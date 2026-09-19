#!/usr/bin/env python3
"""Train LightGBM regressors for OSRM distance/duration and export a single ONNX model.

The exported graph is intentionally simple and dependency-free at runtime:

    input  : features   float32[N, 8]   (see FEATURES below)
    output : distance_m float32[N, 1]   TreeEnsembleRegressor + base offset
    output : duration_s float32[N, 1]   TreeEnsembleRegressor + base offset

Both targets live in one graph so the server performs a single inference pass.

Usage:
    uv run train_export_onnx.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]

# Feature contract shared with server/Program.cs -- order matters.
FEATURES = [
    "orig_lat",
    "orig_lon",
    "dest_lat",
    "dest_lon",
    "haversine_dist_m",
    "bearing_deg",
    "lat_delta",
    "lon_delta",
]

EARTH_RADIUS_M = 6_371_008.8


def geometric_features(orig_lat, orig_lon, dest_lat, dest_lon) -> np.ndarray:
    """Vectorised geometry. Mirrors RoutingFeatures.Compute() in Program.cs."""
    o_lat = np.radians(orig_lat.astype(np.float64))
    o_lon = np.radians(orig_lon.astype(np.float64))
    d_lat_r = np.radians(dest_lat.astype(np.float64))
    d_lon_r = np.radians(dest_lon.astype(np.float64))

    dlat = d_lat_r - o_lat
    dlon = d_lon_r - o_lon

    hav = np.sin(dlat / 2.0) ** 2 + np.cos(o_lat) * np.cos(d_lat_r) * np.sin(dlon / 2.0) ** 2
    haversine = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(o_lat) * np.sin(d_lat_r) - np.sin(o_lat) * np.cos(d_lat_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x)) % 360.0

    return np.column_stack(
        [
            orig_lat,
            orig_lon,
            dest_lat,
            dest_lon,
            haversine,
            bearing,
            dest_lat - orig_lat,
            dest_lon - orig_lon,
        ]
    ).astype(np.float32)


# --------------------------------------------------------------------------------------
# LightGBM -> ONNX TreeEnsembleRegressor
# --------------------------------------------------------------------------------------
def floor_float32(x: float) -> float:
    """Largest float32 that is <= x.

    ONNX `TreeEnsembleRegressor` stores split thresholds as float32 while
    LightGBM compares in float64, and the input features arrive as float32.
    Rounding the threshold *to nearest* can round it *up* past the true
    threshold, so an input between the two takes the wrong branch. Rounding down
    makes `x_f32 <= thr_f32` equivalent to `x_f64 <= thr_f64` for every float32
    x, i.e. the ONNX tree reproduces LightGBM exactly.
    """
    f = np.float32(x)
    if float(f) > x:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def _flatten_tree(tree_structure):
    """Pre-order flatten a LightGBM dump_model tree into a unique node-id space."""
    order: list[dict] = []

    def visit(node) -> int:
        nid = len(order)
        entry = {"leaf": False, "feature": 0, "mode": "LEAF", "value": 0.0, "true": 0, "false": 0, "track": 0, "weight": 0.0}
        order.append(entry)
        if "leaf_value" in node:
            entry["leaf"] = True
            entry["weight"] = float(node["leaf_value"])
        else:
            entry["feature"] = int(node["split_feature"])
            entry["value"] = floor_float32(float(node["threshold"]))
            decision = str(node.get("decision_type", "<="))
            if decision.startswith("<="):
                entry["mode"] = "BRANCH_LEQ"
            elif decision.startswith("=="):
                entry["mode"] = "BRANCH_EQ"
            else:
                raise ValueError(f"unsupported decision_type: {decision}")
            entry["track"] = 1 if node.get("default_left", True) else 0
            entry["true"] = visit(node["left_child"])
            entry["false"] = visit(node["right_child"])
        return nid

    visit(tree_structure)
    return order


def booster_attributes(booster: lgb.Booster) -> dict:
    """Convert a LightGBM booster into TreeEnsembleRegressor attribute lists."""
    dump = booster.dump_model()
    if dump.get("average_output"):
        raise ValueError("average_output models are not supported")

    attrs = {
        "nodes_treeids": [], "nodes_nodeids": [], "nodes_featureids": [], "nodes_modes": [],
        "nodes_values": [], "nodes_truenodeids": [], "nodes_falsenodeids": [],
        "nodes_missing_value_tracks_true": [], "nodes_hitrates": [],
        "target_treeids": [], "target_nodeids": [], "target_ids": [], "target_weights": [],
    }

    for tree_id, tree in enumerate(dump["tree_info"]):
        for node_id, entry in enumerate(_flatten_tree(tree["tree_structure"])):
            attrs["nodes_treeids"].append(tree_id)
            attrs["nodes_nodeids"].append(node_id)
            attrs["nodes_featureids"].append(entry["feature"])
            attrs["nodes_modes"].append(entry["mode"])
            attrs["nodes_values"].append(entry["value"])
            attrs["nodes_truenodeids"].append(entry["true"])
            attrs["nodes_falsenodeids"].append(entry["false"])
            attrs["nodes_missing_value_tracks_true"].append(entry["track"])
            attrs["nodes_hitrates"].append(0.0)
            if entry["leaf"]:
                attrs["target_treeids"].append(tree_id)
                attrs["target_nodeids"].append(node_id)
                attrs["target_ids"].append(0)
                attrs["target_weights"].append(entry["weight"])
    return attrs


def make_tree_node(name: str, attrs: dict, base_value: float, input_name: str) -> onnx.NodeProto:
    return helper.make_node(
        "TreeEnsembleRegressor",
        inputs=[input_name],
        outputs=[name],
        domain="ai.onnx.ml",
        n_targets=1,
        post_transform="NONE",
        base_values=[float(base_value)],
        **attrs,
    )


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        producer_name="testing-ml-tte",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("ai.onnx.ml", 3)],
    )
    # onnx>=1.23 defaults to IR version 14, which onnxruntime (and the C#
    # Microsoft.ML.OnnxRuntime bindings) reject. Opset 21 only needs IR 10.
    model.ir_version = 10
    return model


def build_onnx(boosters: dict[str, lgb.Booster], base_values: dict[str, float]) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(FEATURES)])
    outputs = [
        helper.make_tensor_value_info("distance_m", TensorProto.FLOAT, [None, 1]),
        helper.make_tensor_value_info("duration_s", TensorProto.FLOAT, [None, 1]),
    ]
    nodes = [
        make_tree_node("distance_m", booster_attributes(boosters["distance_m"]), base_values["distance_m"], "features"),
        make_tree_node("duration_s", booster_attributes(boosters["duration_s"]), base_values["duration_s"], "features"),
    ]
    graph = helper.make_graph(nodes, "osrm_approximation", [x], outputs)
    model = _make_model(graph)
    onnx.checker.check_model(model)
    return model


def calibrate_base_value(booster, attrs, features_sample, name: str) -> float:
    """TreeEnsembleRegressor has no notion of LightGBM's boost-from-average
    initial score, so measure it empirically as raw_score - leaf_sum."""
    node = make_tree_node("out", attrs, 0.0, "features")
    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(FEATURES)])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 1])
    model = _make_model(helper.make_graph([node], "calib", [x], [y]))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(["out"], {"features": features_sample})[0].ravel()
    lgb_pred = booster.predict(features_sample, raw_score=True).ravel()
    offsets = lgb_pred - onnx_pred
    # ONNX stores split thresholds as float32 while LightGBM compares in float64,
    # so a feature value sitting exactly on a threshold can take a different
    # branch. That flips a handful of rows to a neighbouring leaf and perturbs
    # their offset, so the offset is only *approximately* constant. Use the
    # median (robust to those few rows) and only fail if a large fraction is off,
    # which is what happens when the leaf values themselves are wrong.
    base = float(np.median(offsets))
    close = float(np.mean(np.abs(offsets - base) <= 0.5))
    if close < 0.90:
        raise RuntimeError(
            f"base offset is not constant (only {close * 100:.1f}% of rows within 0.5 of median "
            f"{base:.4f}); export would be wrong"
        )
    if close < 1.0:
        print(f"[export]   note: {name} {100 * (1 - close):.2f}% of rows off the median "
              f"(float32 threshold rounding)")
    return base


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(y_true > 1e-6, err / y_true, np.nan)
    return {
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
        "medape_pct": float(np.nanmedian(rel) * 100.0),
        "p90_abs_err": float(np.percentile(err, 90)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


BUCKETS = [(0, 1_000, "<1km"), (1_000, 3_000, "1-3km"), (3_000, 10_000, "3-10km"),
           (10_000, 25_000, "10-25km"), (25_000, float("inf"), ">25km")]


def bucket_report(haversine: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Error broken down by straight-line separation -- the headline MAE hides
    the fact that long trips dominate the absolute numbers."""
    out = {}
    for lo, hi, label in BUCKETS:
        mask = (haversine >= lo) & (haversine < hi)
        if not mask.any():
            continue
        m = metrics(y_true[mask], y_pred[mask])
        m["n"] = int(mask.sum())
        out[label] = m
    return out


def inverse_frequency_weights(dist_m: np.ndarray) -> np.ndarray:
    """E1 (see IMPROVEMENTS.md): up-weight the short-separation buckets that the
    all-pairs training set barely contains (0.23% are < 1 km, median 17.3 km).
    Each distance bucket gets equal total weight, normalised to mean 1 so that
    the effective learning rate is unchanged."""
    w = np.ones(len(dist_m), dtype=np.float64)
    for lo, hi, _label in BUCKETS:
        mask = (dist_m >= lo) & (dist_m < hi)
        if mask.any():
            w[mask] = 1.0 / mask.sum()
    w *= len(w) / w.sum()
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default=str(ROOT / "data" / "processed" / "samples.parquet"))
    ap.add_argument("--extra-samples", default=None,
                    help="E5: extra parquet rows (e.g. off-network pairs) appended to the training pool. "
                         "Defaults to data/processed/offnetwork.parquet when that file exists.")
    ap.add_argument("--out", default=str(ROOT / "server" / "models" / "model.onnx"))
    ap.add_argument("--num-leaves", type=int, default=511)
    ap.add_argument("--estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--weighting", choices=["none", "inv_freq"], default="inv_freq",
                    help="E1: inverse-frequency sample weights over the training separation buckets")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    print(f"[train] loading {args.samples}")
    df = pd.read_parquet(args.samples)
    print(f"[train] {len(df):,} pairs")
    extra_path = args.extra_samples
    if extra_path is None:
        default_extra = ROOT / "data" / "processed" / "offnetwork.parquet"
        if default_extra.exists():
            extra_path = str(default_extra)
    if extra_path:
        extra = pd.read_parquet(extra_path)
        print(f"[train] + {len(extra):,} extra pairs from {extra_path}")
        df = pd.concat([df, extra], ignore_index=True)

    X = geometric_features(
        df["orig_lat"].to_numpy(), df["orig_lon"].to_numpy(), df["dest_lat"].to_numpy(), df["dest_lon"].to_numpy()
    )
    targets = {"distance_m": df["osrm_distance_m"].to_numpy(np.float32), "duration_s": df["osrm_duration_s"].to_numpy(np.float32)}

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_test = max(1, int(len(X) * args.test_frac))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_tr, X_te = X[train_idx], X[test_idx]

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.estimators,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "verbose": -1,
    }
    if args.threads:
        params["num_threads"] = args.threads

    # E1 reweighting: same weights for both targets (driven by road distance).
    sample_weight = None
    if args.weighting == "inv_freq":
        sample_weight = inverse_frequency_weights(df["osrm_distance_m"].to_numpy(np.float64)[train_idx])
        print(f"[train] inverse-frequency weights: {args.weighting} "
              f"(min={sample_weight.min():.4f} max={sample_weight.max():.2f})")

    boosters: dict[str, lgb.Booster] = {}
    report: dict[str, dict] = {}

    # Record the extra source as a repo-relative path (keep host paths out of the metadata).
    try:
        extra_rel = str(Path(extra_path).resolve().relative_to(ROOT)) if extra_path else None
    except ValueError:
        extra_rel = Path(extra_path).name if extra_path else None

    for name, y in targets.items():
        print(f"\n[train] === {name} ===")
        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y[train_idx], feature_name=FEATURES, sample_weight=sample_weight)
        booster = model.booster_
        boosters[name] = booster
        print(f"[train] fit in {time.time() - t0:.1f}s ({booster.num_trees()} trees)")

        pred = booster.predict(X_te)
        m = metrics(y[test_idx], pred)
        report[name] = m
        print(
            f"[train] {name}: MAE={m['mae']:,.1f} MedAE={m['medae']:,.1f} "
            f"MedAPE={m['medape_pct']:.1f}% p90|err|={m['p90_abs_err']:,.1f} RMSE={m['rmse']:,.1f}"
        )

    # Haversine baseline for distance gives the accuracy numbers context.
    hav = X_te[:, FEATURES.index("haversine_dist_m")]
    base = metrics(targets["distance_m"][test_idx], hav)
    print(f"\n[baseline] haversine-as-distance: MAE={base['mae']:,.1f} MedAE={base['medae']:,.1f} MedAPE={base['medape_pct']:.1f}%")

    print("\n[export] calibrating LightGBM boost-from-average offsets")
    base_values = {}
    for name, booster in boosters.items():
        base_values[name] = calibrate_base_value(booster, booster_attributes(booster), X_tr[:4096], name)
        print(f"[export] {name}: base_value={base_values[name]:.6f}")

    model = build_onnx(boosters, base_values)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # End-to-end ONNX verification against LightGBM on held-out rows.
    print("[verify] comparing ONNX Runtime vs LightGBM on test set")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(["distance_m", "duration_s"], {"features": X_te})
    verification = {}
    buckets = {}
    hav_te = X_te[:, FEATURES.index("haversine_dist_m")]
    for name, idx in (("distance_m", 0), ("duration_s", 1)):
        onnx_pred = got[idx].ravel()
        lgb_pred = boosters[name].predict(X_te)
        diff = np.abs(onnx_pred - lgb_pred)
        rel = diff / np.maximum(np.abs(lgb_pred), 1e-6)
        # ONNX stores split thresholds as float32 while LightGBM compares in
        # float64, so a feature value sitting exactly on a split point can take
        # a different branch. A handful of such rows is expected and immaterial;
        # a systematic mismatch is not, so require 99.9% agreement within
        # 1 unit or 1%, whichever is looser.
        tol = np.maximum(1.0, 0.01 * np.abs(lgb_pred))
        within = float((diff <= tol).mean())
        worst = int(np.argmax(diff))
        verification[name] = {
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float(rel.max()),
            "fraction_within_1pct": within,
            "worst_row": {
                "lgb": float(lgb_pred[worst]),
                "onnx": float(onnx_pred[worst]),
            },
        }
        print(
            f"[verify] {name}: max|ONNX-LGBM|={diff.max():.4f} max_rel={rel.max():.2e} "
            f"within 1%/1unit = {within * 100:.4f}%"
        )
        if within < 0.999:
            raise RuntimeError(f"ONNX export diverges from LightGBM for {name}: only {within * 100:.3f}% agree")

        m = metrics(targets[name][test_idx], onnx_pred)
        print(
            f"[verify] {name}: ONNX MAE={m['mae']:,.1f} MedAE={m['medae']:,.1f} MedAPE={m['medape_pct']:.1f}%"
        )
        report[name]["onnx"] = m
        buckets[name] = bucket_report(hav_te, targets[name][test_idx], onnx_pred)
        print(f"[verify] {name} by straight-line separation:")
        for label, bm in buckets[name].items():
            print(
                f"[verify]   {label:>8}: n={bm['n']:>8,} MAE={bm['mae']:>9,.1f} "
                f"MedAE={bm['medae']:>8,.1f} MedAPE={bm['medape_pct']:>5.1f}%"
            )

    meta = {
        "features": FEATURES,
        "outputs": ["distance_m", "duration_s"],
        "base_values": base_values,
        "params": params,
        "weighting": args.weighting,
        "extra_samples": extra_rel,
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "metrics_test": report,
        "by_separation": buckets,
        "haversine_baseline_distance_m": base,
        "verification": verification,
        "model_bytes": out_path.stat().st_size,
    }
    meta_path = out_path.with_name("model_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
