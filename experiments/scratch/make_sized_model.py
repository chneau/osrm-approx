#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
# ]
# ///
"""Write a structurally valid `model.bin` with a target LightGBM leaf count.

Used to measure RSS on the capacity curve without paying for two more full trainings
plus ONNX exports just to learn how much memory a given table size costs.

The server's loader does not care about tree *semantics* — it reads the same arrays and
allocates the same typed buffers, whose sizes are a pure function of (n_trees, n_nodes).
So a table with the right shape costs exactly the same RSS as a real trained model of
that shape (predictions are meaningless; leaf values are zero).

Shape facts this relies on, both verifiable against the two real exports:
  * `num_leaves = L` -> `2L - 1` nodes per tree (63 -> 125, 511 -> 1023)
  * 400 trees per target (n_estimators=400), 2 targets
  * 22 B per node in the file: feature 4 + threshold 4 + left 4 + right 4 + value 4
    + is_leaf 1 + default_left 1

Self-check: run it at 63 and 511 and confirm the synthetic files have ~the same size as
the real `model_small.bin` / `model.bin`, and that the measured RSS matches. If the
endpoints agree, the interpolated 127/255 points are trustworthy.

Usage:
    uv run experiments/scratch/make_sized_model.py --leaves 63 127 255 511
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "scratch"
N_TREES = 400
N_TARGETS = 2
N_FEATURES = 8


def build(leaves: int) -> dict:
    """Emit one target's node block with the given leaf count."""
    nodes_per_tree = 2 * leaves - 1
    n_nodes = N_TREES * nodes_per_tree

    offsets = np.arange(N_TREES + 1, dtype=np.int32) * nodes_per_tree
    feature = np.full(n_nodes, -1, dtype=np.int32)
    threshold = np.zeros(n_nodes, dtype=np.float32)
    left = np.zeros(n_nodes, dtype=np.int32)
    right = np.zeros(n_nodes, dtype=np.int32)
    value = np.zeros(n_nodes, dtype=np.float32)
    is_leaf = np.ones(n_nodes, dtype=np.uint8)

    # Each tree: node 0 is an internal split whose children are leaves, so the walk
    # terminates immediately. Node count (and therefore the allocated arrays) is exactly
    # what a real num_leaves=L model would have.
    for t in range(N_TREES):
        base = t * nodes_per_tree
        is_leaf[base] = 0
        feature[base] = 0
        threshold[base] = 0.0
        left[base] = base + 1
        right[base] = base + 2
    left[is_leaf == 1] = np.arange(n_nodes, dtype=np.int32)[is_leaf == 1]
    right[is_leaf == 1] = np.arange(n_nodes, dtype=np.int32)[is_leaf == 1]

    return dict(n_trees=N_TREES, n_nodes=n_nodes, offsets=offsets, feature=feature,
                threshold=threshold, left=left, right=right, value=value,
                is_leaf=is_leaf, default_left=np.zeros(n_nodes, dtype=np.uint8))


def write(path: Path, leaves: int) -> int:
    with path.open("wb") as f:
        f.write(b"OSRT")
        f.write(struct.pack("<III", 1, N_FEATURES, N_TARGETS))
        for _ in range(N_TARGETS):
            t = build(leaves)
            f.write(struct.pack("<f", 0.0))
            f.write(struct.pack("<II", t["n_trees"], t["n_nodes"]))
            t["offsets"].astype("<i4").tofile(f)
            t["feature"].astype("<i4").tofile(f)
            t["threshold"].astype("<f4").tofile(f)
            t["left"].astype("<i4").tofile(f)
            t["right"].astype("<i4").tofile(f)
            t["value"].astype("<f4").tofile(f)
            t["is_leaf"].astype("<u1").tofile(f)
            t["default_left"].astype("<u1").tofile(f)
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaves", type=int, nargs="+", default=[63, 127, 255, 511])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    for leaves in args.leaves:
        path = args.out_dir / f"model_sized_{leaves}leaf.bin"
        size = write(path, leaves)
        nodes = N_TREES * (2 * leaves - 1)
        print(f"[sized] L={leaves:4d} -> {nodes:>8,} nodes/target  {size / 1e6:6.2f} MB  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
