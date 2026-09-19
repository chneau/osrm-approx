#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "onnx>=1.16",
# ]
# ///
"""E11 (see IMPROVEMENTS.md): compile the ONNX tree ensemble to a compact binary.

Reads the TreeEnsembleRegressor nodes out of `server/models/model.onnx` and
writes `server/models/model.bin` in a flat, loader-friendly layout so the server
can evaluate the trees with a tiny custom interpreter instead of ONNX Runtime
(which is the dominant RSS cost). No retraining: this is a pure re-serialisation
of the already-verified model.

Binary layout (little-endian):

    magic       char[4]  "OSRT"
    version     uint32   1
    n_features  uint32
    n_targets   uint32
    per target (in graph order: distance_m, duration_s):
        base_value    float32
        n_trees       uint32
        n_nodes       uint32
        tree_offsets  int32[n_trees+1]   start index of each tree's node block
        feature       int32[n_nodes]
        threshold     float32[n_nodes]
        left          int32[n_nodes]     global row of x<=threshold child
        right         int32[n_nodes]     global row of x>threshold child
        value         float32[n_nodes]   leaf value (0 for internal nodes)
        is_leaf       uint8[n_nodes]
        default_left  uint8[n_nodes]

Usage:
    uv run export_binary.py [--onnx server/models/model.onnx] [--out server/models/model.bin]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"OSRT"
VERSION = 1


def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: a for a in node.attribute}


def _ints(a: onnx.AttributeProto) -> list[int]:
    return list(a.ints)


def _floats(a: onnx.AttributeProto) -> list[float]:
    return list(a.floats)


def _strings(a: onnx.AttributeProto) -> list[str]:
    return [s.decode() for s in a.strings]


def extract(node: onnx.NodeProto) -> dict:
    """Pull one TreeEnsembleRegressor into flat per-node arrays."""
    at = _attrs(node)
    treeids = np.asarray(_ints(at["nodes_treeids"]), dtype=np.int64)
    nodeids = np.asarray(_ints(at["nodes_nodeids"]), dtype=np.int64)
    feature = np.asarray(_ints(at["nodes_featureids"]), dtype=np.int32)
    modes = _strings(at["nodes_modes"])
    thresholds = np.asarray(_floats(at["nodes_values"]), dtype=np.float32)
    truenodes = np.asarray(_ints(at["nodes_truenodeids"]), dtype=np.int32)
    falsenodes = np.asarray(_ints(at["nodes_falsenodeids"]), dtype=np.int32)
    tracks = np.asarray(_ints(at["nodes_missing_value_tracks_true"]), dtype=np.uint8)
    base = float(_floats(at["base_values"])[0])
    n_nodes = len(treeids)

    # Node blocks are emitted per tree, in tree order, with per-tree ids 0..k-1.
    n_trees = int(treeids.max()) + 1
    tree_offsets = np.zeros(n_trees + 1, dtype=np.int32)
    for t in range(n_trees):
        tree_offsets[t] = int(np.argmax(treeids == t)) if (treeids == t).any() else n_nodes
    tree_offsets[n_trees] = n_nodes

    # Sanity: the flattened order must be (tree, nodeid) ascending so that a
    # per-tree node id maps to offset[tree] + nodeid.
    for t in range(n_trees):
        lo, hi = tree_offsets[t], tree_offsets[t + 1]
        if not np.array_equal(treeids[lo:hi], np.full(hi - lo, t)):
            raise ValueError(f"tree {t} nodes are not contiguous")
        if not np.array_equal(nodeids[lo:hi], np.arange(hi - lo)):
            raise ValueError(f"tree {t} node ids are not 0..k-1 in order")

    # Child ids in the ONNX attributes are *per-tree* node ids (0..k-1). Rebase
    # them to global node rows (tree_offsets[t] + nodeid) so the interpreter can
    # walk `left[i]`/`right[i]` directly. Leaf rows keep a self-reference since
    # the walk never dereferences them (guarded by is_leaf).
    left = np.zeros(n_nodes, dtype=np.int32)
    right = np.zeros(n_nodes, dtype=np.int32)
    is_leaf = np.array([m == "LEAF" for m in modes], dtype=np.uint8)
    for t in range(n_trees):
        lo, hi = int(tree_offsets[t]), int(tree_offsets[t + 1])
        left[lo:hi] = tree_offsets[t] + truenodes[lo:hi]
        right[lo:hi] = tree_offsets[t] + falsenodes[lo:hi]
    left[is_leaf == 1] = np.arange(n_nodes, dtype=np.int32)[is_leaf == 1]
    right[is_leaf == 1] = np.arange(n_nodes, dtype=np.int32)[is_leaf == 1]
    value = np.zeros(n_nodes, dtype=np.float32)

    # Leaf values: target_weights keyed by (target_treeids, target_nodeids).
    ttree = np.asarray(_ints(at["target_treeids"]), dtype=np.int64)
    tnode = np.asarray(_ints(at["target_nodeids"]), dtype=np.int64)
    tweights = np.asarray(_floats(at["target_weights"]), dtype=np.float32)
    for k in range(len(ttree)):
        gidx = int(tree_offsets[ttree[k]]) + int(tnode[k])
        value[gidx] = tweights[k]

    return dict(
        base=base, n_trees=n_trees, n_nodes=n_nodes, tree_offsets=tree_offsets,
        feature=feature, threshold=thresholds, left=left, right=right,
        value=value, is_leaf=is_leaf, default_left=tracks,
    )


def write(out: Path, targets: list[dict], n_features: int) -> None:
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, n_features, len(targets)))
        for t in targets:
            f.write(struct.pack("<f", t["base"]))
            f.write(struct.pack("<II", t["n_trees"], t["n_nodes"]))
            t["tree_offsets"].astype("<i4").tofile(f)
            t["feature"].astype("<i4").tofile(f)
            t["threshold"].astype("<f4").tofile(f)
            t["left"].astype("<i4").tofile(f)
            t["right"].astype("<i4").tofile(f)
            t["value"].astype("<f4").tofile(f)
            t["is_leaf"].astype("<u1").tofile(f)
            t["default_left"].astype("<u1").tofile(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default=str(ROOT / "server" / "models" / "model.onnx"))
    ap.add_argument("--out", default=str(ROOT / "server" / "models" / "model.bin"))
    args = ap.parse_args()

    model = onnx.load(args.onnx, load_external_data=False)
    tree_nodes = [n for n in model.graph.node if n.op_type == "TreeEnsembleRegressor"]
    if len(tree_nodes) != 2:
        raise SystemExit(f"expected 2 TreeEnsembleRegressor nodes, found {len(tree_nodes)}")
    # Preserve graph output order (distance_m, duration_s) so the loader's target
    # indices match the server's Predict() contract.
    order = ["distance_m", "duration_s"]
    by_out = {n.output[0]: n for n in tree_nodes}
    targets = [extract(by_out[name]) for name in order]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, targets, n_features=8)
    print(f"[binary] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    for name, t in zip(order, targets):
        print(f"[binary]   {name}: {t['n_trees']} trees, {t['n_nodes']:,} nodes, base={t['base']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
