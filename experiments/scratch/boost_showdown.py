#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "catboost>=1.2.10",
#   "lightgbm>=4.7.0",
#   "numpy>=2.5.3",
#   "onnx>=1.23.0",
#   "onnxruntime>=1.30.0",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "scikit-learn>=1.9.1",
#   "xgboost>=3.4.1",
# ]
# ///
"""Three questions in one run, on one identical split (see IMPROVEMENTS.md).

Protocol is exactly `price_capacity.py` / `experiments/try_s1_fair.py`: train on
`samples.parquet` + 80% of `offnetwork.parquet` with E1 inverse-frequency weights, score
on the unseen 20% (31,761 raw-coordinate pairs). Every model sees the same rows, the same
weights and the same held-out set, so the rows are comparable.

1. **Accuracy per byte (LightGBM).** The capacity sweep only varied `num_leaves`. The
   thing that now matters is how much `model.bin` we must ship for a given error, so this
   varies leaves x trees x learning rate and reports the *artifact size* next to the error.
   Artifact size is exact arithmetic, not an estimate: 2 targets x trees x (2*leaves - 1)
   nodes x 22 B, the layout `export_binary.py` writes (validated against both real
   artifacts to within 0.002%).

2. **XGBoost.** `grow_policy=lossguide` + `max_leaves=511` is the like-for-like arm
   (same tree shape, same objective); `depthwise` + `max_depth=6` is the "XGBoost defaults
   are more forgiving" arm from the generic advice. `min_child_weight=40` matches
   LightGBM's `min_child_samples=40` because squared-error hessians are 1 per row.

3. **CatBoost.** Its differentiator (categorical handling) is void here — all 8 features
   are numeric — so the interesting arm is the *shape*: symmetric/oblivious trees at
   `depth=9` (512 leaves, the same node budget as LightGBM's 511) and `depth=6`. Oblivious
   trees are what would let the C# interpreter index leaves with a bitmask instead of
   walking, so this also prices the serving simplification.

Usage:
    uv run experiments/scratch/boost_showdown.py
    uv run experiments/scratch/boost_showdown.py --only lgbm xgb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from train_export_onnx import geometric_features, inverse_frequency_weights  # noqa: E402

import lightgbm as lgb  # noqa: E402

PROC = ROOT / "data" / "processed"
BUCKETS = [
    ("<1km", 0.0, 1_000.0),
    ("1-3km", 1_000.0, 3_000.0),
    ("3-10km", 3_000.0, 10_000.0),
    ("10-25km", 10_000.0, 25_000.0),
    (">25km", 25_000.0, float("inf")),
]
BYTES_PER_NODE = 22  # export_binary.py layout, verified against the shipped artifacts

# (name, leaves, trees, lr). Node budget = trees * (2*leaves - 1); artifact = 2 budgets * 22 B.
LGBM_CONFIGS = [
    ("lgbm-511L-400t-lr08 (shipped)", 511, 400, 0.08),
    ("lgbm-1023L-200t-lr05", 1023, 200, 0.05),
    ("lgbm-255L-800t-lr05", 255, 800, 0.05),
    ("lgbm-511L-200t-lr05", 511, 200, 0.05),
    ("lgbm-255L-400t-lr05", 255, 400, 0.05),
    ("lgbm-255L-200t-lr05", 255, 200, 0.05),
]


def artifact_mb(leaves: int, trees: int) -> float:
    return 2 * trees * (2 * leaves - 1) * BYTES_PER_NODE / 1e6


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(truth - pred)
    rel = np.where(truth > 1e-6, err / truth, np.nan) * 100.0
    return {
        "n": int(len(truth)),
        "medape": float(np.nanmedian(rel)),
        "mape": float(np.nanmean(rel)),
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
    }


def score(pred_d, pred_t, truth_d, truth_t) -> dict:
    out = {
        "distance": {"overall": metrics(truth_d, pred_d), "by_bucket": {}},
        "duration": {"overall": metrics(truth_t, pred_t), "by_bucket": {}},
    }
    for key, truth, pred in (("distance", truth_d, pred_d), ("duration", truth_t, pred_t)):
        for label, lo, hi in BUCKETS:
            mask = (truth_d >= lo) & (truth_d < hi)
            if mask.sum() >= 30:
                out[key]["by_bucket"][label] = metrics(truth[mask], pred[mask])
    return out


def fit_lgbm(X, y_d, y_t, w, X_te, leaves, trees, lr):
    params = dict(objective="regression", metric="mae", num_leaves=leaves, learning_rate=lr,
                  n_estimators=trees, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, verbose=-1, num_threads=16)
    preds = {}
    for name, y in (("distance", y_d), ("duration", y_t)):
        preds[name] = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w).predict(X_te)
    return preds


def fit_xgb(X, y_d, y_t, w, X_te, grow_policy, max_leaves, max_depth, trees, lr):
    import xgboost as xgb

    params = dict(objective="reg:squarederror", tree_method="hist", grow_policy=grow_policy,
                  n_estimators=trees, learning_rate=lr, subsample=0.8, colsample_bytree=0.9,
                  min_child_weight=40, n_jobs=16, verbosity=0, random_state=0)
    if grow_policy == "lossguide":
        params["max_leaves"] = max_leaves
        params["max_depth"] = 0
    else:
        params["max_depth"] = max_depth
    preds = {}
    for name, y in (("distance", y_d), ("duration", y_t)):
        preds[name] = xgb.XGBRegressor(**params).fit(X, y, sample_weight=w).predict(X_te)
    return preds


def fit_cat(X, y_d, y_t, w, X_te, depth, trees, lr):
    from catboost import CatBoostRegressor

    model_kw = dict(iterations=trees, depth=depth, learning_rate=lr, loss_function="RMSE",
                    thread_count=16, verbose=0, random_seed=0, allow_writing_files=False)
    preds = {}
    for name, y in (("distance", y_d), ("duration", y_t)):
        model = CatBoostRegressor(**model_kw).fit(X, y, sample_weight=w)
        preds[name] = model.predict(X_te)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=["lgbm", "xgb", "cat"],
                    choices=["lgbm", "xgb", "cat"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments" / "scratch" / "boost_showdown.json")
    args = ap.parse_args()

    t0 = time.time()
    samples = pd.read_parquet(PROC / "samples.parquet")
    off = pd.read_parquet(PROC / "offnetwork.parquet")
    rng = np.random.default_rng(0)  # same split as price_capacity.py / try_s1_fair.py
    perm = rng.permutation(len(off))
    n_te = len(off) // 5
    te = off.iloc[perm[:n_te]].reset_index(drop=True)
    tr_off = off.iloc[perm[n_te:]].reset_index(drop=True)

    truth_d = te["osrm_distance_m"].to_numpy(np.float64)
    truth_t = te["osrm_duration_s"].to_numpy(np.float64)

    both = pd.concat([samples, tr_off], ignore_index=True)
    X = geometric_features(both["orig_lat"].to_numpy(), both["orig_lon"].to_numpy(),
                           both["dest_lat"].to_numpy(), both["dest_lon"].to_numpy())
    w = inverse_frequency_weights(both["osrm_distance_m"].to_numpy(np.float64))
    X_te = geometric_features(te["orig_lat"].to_numpy(), te["orig_lon"].to_numpy(),
                              te["dest_lat"].to_numpy(), te["dest_lon"].to_numpy())
    y_d = both["osrm_distance_m"].to_numpy(np.float32)
    y_t = both["osrm_duration_s"].to_numpy(np.float32)
    print(f"[showdown] train={len(both):,} x {X.shape[1]}  test={len(te):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    results: dict[str, dict] = {}
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def record(name: str, preds: dict, secs: float, leaves: int, trees: int, extra: dict):
        rec = score(preds["distance"], preds["duration"], truth_d, truth_t)
        rec.update({"leaves": leaves, "trees": trees, "fit_seconds": round(secs, 1),
                    "artifact_mb": round(artifact_mb(leaves, trees), 2), **extra})
        results[name] = rec
        out.write_text(json.dumps(results, indent=2) + "\n")  # checkpoint per config
        d, u = rec["distance"]["overall"], rec["duration"]["overall"]
        print(f"[showdown] {name:34s} {rec['artifact_mb']:6.2f} MB  "
              f"dist {d['medape']:5.2f}% {d['mae']:7.0f}m  dur {u['medape']:5.2f}% {u['mae']:5.0f}s "
              f"({secs:.0f}s)", flush=True)

    if "lgbm" in args.only:
        for name, leaves, trees, lr in LGBM_CONFIGS:
            print(f"[showdown] lightgbm {name} ...", flush=True)
            ts = time.time()
            record(name, fit_lgbm(X, y_d, y_t, w, X_te, leaves, trees, lr), time.time() - ts,
                   leaves, trees, {"library": "lightgbm", "learning_rate": lr})

    if "xgb" in args.only:
        import xgboost  # noqa: F401
        for name, policy, max_leaves, max_depth in (
            ("xgb-lossguide-511L-400t-lr08", "lossguide", 511, 0),
            ("xgb-depthwise-depth6-400t-lr08", "depthwise", 0, 6),
        ):
            print(f"[showdown] {name} ...", flush=True)
            ts = time.time()
            preds = fit_xgb(X, y_d, y_t, w, X_te, policy, max_leaves, max_depth, 400, 0.08)
            leaves = 511 if policy == "lossguide" else 2 ** 6 - 1
            record(name, preds, time.time() - ts, leaves, 400,
                   {"library": "xgboost", "grow_policy": policy, "learning_rate": 0.08})

    if "cat" in args.only:
        import catboost  # noqa: F401
        for name, depth in (("cat-symmetric-depth9-400t-lr08", 9),
                            ("cat-symmetric-depth6-400t-lr08", 6)):
            print(f"[showdown] {name} ...", flush=True)
            ts = time.time()
            preds = fit_cat(X, y_d, y_t, w, X_te, depth, 400, 0.08)
            record(name, preds, time.time() - ts, 2 ** depth - 1, 400,
                   {"library": "catboost", "depth": depth, "learning_rate": 0.08})

    print("\n=== distance / duration MedAPE vs artifact size (held-out raw coordinates) ===")
    print(f"{'config':34s} {'library':9s} {'MB':>7s} {'dist MedAPE':>12s} {'dist MAE':>9s} "
          f"{'dur MedAPE':>11s} {'dur MAE':>8s}")
    for name, rec in sorted(results.items(), key=lambda kv: kv[1]["artifact_mb"]):
        d, u = rec["distance"]["overall"], rec["duration"]["overall"]
        print(f"{name:34s} {rec['library']:9s} {rec['artifact_mb']:7.2f} {d['medape']:11.2f}% "
              f"{d['mae']:9.0f} {u['medape']:10.2f}% {u['mae']:8.0f}")
    print(f"\n[showdown] wrote {out}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
