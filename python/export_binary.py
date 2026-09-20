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

After E11 the loader was rewritten to *stream* the file, which made the model cost
~1.0x its artifact size in RSS instead of ~1.9x. That also removed the only reason
the layout was wide: with stock ONNX Runtime, narrow types were impossible (E10), but
the interpreter is ours, so v2 packs the tables to the widths the data actually needs:

    feature     uint8    features are indices 0..n_features-1
    leaf flag   implicit `left < 0` marks a leaf, so the old is_leaf byte is gone
    value       float32 or uint16   uint16 stores (v / scale) + 32768 per target

Layout (little-endian):

    magic       char[4]  "OSRT"
    version     uint32   2
    flags       uint32   bit0 = leaf values are uint16-quantised
    n_features  uint32
    n_targets   uint32
    per target (in graph order: distance_m, duration_s):
        base_value    float32
        value_scale   float32           decode: (q - 32768) * value_scale
        n_trees       uint32
        n_nodes       uint32
        tree_offsets  int32[n_trees+1]  start index of each tree's node block
        feature       uint8[n_nodes]
        threshold     float32[n_nodes]
        left          int32[n_nodes]    internal: row of the x<=threshold child;
                                        leaf: ~row (negative), i.e. is_leaf == left < 0
        right         int32[n_nodes]    internal: row of the x>threshold child
        value         float32[n_nodes]  (flags bit0 clear)
        value_q       uint16[n_nodes]   (flags bit0 set)

`--value-encoding u16` is the aggressive arm: it quantises leaf values only and leaves
every split threshold exact, so it can never send an input down a different branch than
LightGBM would (the failure mode behind the E7 export bug). Measure the accuracy delta
before adopting it — `--value-encoding f32` is lossless and bit-identical to v1 output.

Usage:
    uv run export_binary.py [--onnx server/models/model.onnx] [--out server/models/model.bin]
    uv run export_binary.py --value-encoding u16
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
VERSION = 2
FLAG_VALUE_U16 = 1
U16_OFFSET = 32768


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
    feature = np.asarray(_ints(at["nodes_featureids"]), dtype=np.uint8)
    modes = _strings(at["nodes_modes"])
    thresholds = np.asarray(_floats(at["nodes_values"]), dtype=np.float32)
    truenodes = np.asarray(_ints(at["nodes_truenodeids"]), dtype=np.int32)
    falsenodes = np.asarray(_ints(at["nodes_falsenodeids"]), dtype=np.int32)
    base = float(_floats(at["base_values"])[0])
    n_nodes = len(treeids)
    if feature.max(initial=0) > 255:
        raise SystemExit("feature index exceeds uint8; widen the v2 layout")

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

    # Child ids in the ONNX attributes are *per-tree* node ids (0..k-1). Rebase them
    # to global node rows (tree_offsets[t] + nodeid) so the interpreter can walk
    # `left[i]`/`right[i]` directly.
    left = np.zeros(n_nodes, dtype=np.int32)
    right = np.zeros(n_nodes, dtype=np.int32)
    is_leaf = np.array([m == "LEAF" for m in modes], dtype=np.uint8)
    for t in range(n_trees):
        lo, hi = int(tree_offsets[t]), int(tree_offsets[t + 1])
        left[lo:hi] = tree_offsets[t] + truenodes[lo:hi]
        right[lo:hi] = tree_offsets[t] + falsenodes[lo:hi]

    # v2: leaves are marked by a negative `left` (~row) instead of a stored is_leaf
    # byte, and the walk never dereferences their children.
    rows = np.arange(n_nodes, dtype=np.int32)
    leaf = is_leaf == 1
    left[leaf] = ~rows[leaf]
    right[leaf] = ~rows[leaf]
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
        value=value, is_leaf=is_leaf,
    )


def quantise_values(target: dict) -> tuple[np.ndarray, float]:
    """uint16 leaf values with a per-target scale, offset by 32768 so signs survive."""
    leaf = target["is_leaf"] == 1
    magnitude = float(np.abs(target["value"][leaf]).max()) if leaf.any() else 0.0
    scale = magnitude / (U16_OFFSET - 1) if magnitude > 0 else 1.0
    q = np.zeros(target["n_nodes"], dtype=np.uint16)
    scaled = np.rint(target["value"][leaf].astype(np.float64) / scale).astype(np.int64)
    q[leaf] = np.clip(scaled + U16_OFFSET, 0, 65535).astype(np.uint16)
    return q, scale


def write(out: Path, targets: list[dict], n_features: int, use_u16: bool) -> None:
    flags = FLAG_VALUE_U16 if use_u16 else 0
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, flags, n_features, len(targets)))
        for t in targets:
            if use_u16:
                q, scale = quantise_values(t)
            else:
                q, scale = None, 1.0
            f.write(struct.pack("<ffII", t["base"], scale, t["n_trees"], t["n_nodes"]))
            t["tree_offsets"].astype("<i4").tofile(f)
            t["feature"].astype("<u1").tofile(f)
            t["threshold"].astype("<f4").tofile(f)
            t["left"].astype("<i4").tofile(f)
            t["right"].astype("<i4").tofile(f)
            if use_u16:
                q.astype("<u2").tofile(f)
            else:
                t["value"].astype("<f4").tofile(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default=str(ROOT / "server" / "models" / "model.onnx"))
    ap.add_argument("--out", default=str(ROOT / "server" / "models" / "model.bin"))
    ap.add_argument("--value-encoding", choices=["f32", "u16"], default="f32",
                    help="f32 is lossless; u16 quantises leaf values (see docstring)")
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

    use_u16 = args.value_encoding == "u16"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, targets, n_features=8, use_u16=use_u16)
    print(f"[binary] wrote {out} ({out.stat().st_size / 1e6:.2f} MB, "
          f"v{VERSION}, value_encoding={args.value_encoding})")
    for name, t in zip(order, targets):
        detail = f"{t['n_trees']} trees, {t['n_nodes']:,} nodes, base={t['base']:.6f}"
        if use_u16:
            _, scale = quantise_values(t)
            detail += f", value_scale={scale:.6g}"
        print(f"[binary]   {name}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
