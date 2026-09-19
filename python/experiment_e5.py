#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=17.0",
#   "lightgbm>=4.5",
#   "scikit-learn>=1.5",
# ]
# ///
"""E5 (see IMPROVEMENTS.md): does adding off-network training pairs help?

Trains two models per target with the same config (E1 inverse-frequency weights):
  A) grid samples only          (what ships today)
  B) grid + off-network samples (points OSRM had to snap)

and scores both on two held-out sets:
  - a distance-balanced *grid* test set (generalisation to snapped coords)
  - an *off-network* test set (generalisation to raw coords, what the API gets)

    uv run experiment_e5.py --leaves 127 --estimators 200 --train-rows 1500000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import (  # noqa: E402
    BUCKETS,
    ROOT,
    bucket_labels,
    build_features,
    make_weights,
    mae,
    medape,
)

DEFAULT_GRID = ROOT / "data" / "processed" / "samples.parquet"
DEFAULT_OFF = ROOT / "data" / "processed" / "offnetwork.parquet"


def train(df, params, weights):
    X, hav = build_features(df, "base")
    w = make_weights(df["osrm_distance_m"].to_numpy(np.float64), weights)
    boosters = {}
    for key in ("osrm_distance_m", "osrm_duration_s"):
        y = df[key].to_numpy(np.float64)
        boosters[key] = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w).booster_
    return boosters


def evaluate(boosters, df, tag):
    X, _ = build_features(df, "base")
    out = {}
    for key in ("osrm_distance_m", "osrm_duration_s"):
        truth = df[key].to_numpy(np.float64)
        pred = boosters[key].predict(X)
        labels = bucket_labels(None, truth)
        by = {}
        for name, _, _ in BUCKETS:
            m = labels == name
            if m.sum() < 20:
                continue
            by[name] = {"medape": medape(truth[m], pred[m]), "mae": mae(truth[m], pred[m]), "n": int(m.sum())}
        out[key] = {"medape": medape(truth, pred), "mae": mae(truth, pred), "n": int(len(truth)), "by_bucket": by}
    print(f"[e5] {tag}")
    for key, rec in out.items():
        buckets = " ".join(f"{b}:{v['medape']:.0f}%" for b, v in rec["by_bucket"].items())
        print(f"[e5]   {key:16s} MedAPE={rec['medape']:5.1f}% MAE={rec['mae']:>10,.0f}  [{buckets}]")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    ap.add_argument("--off", default=str(DEFAULT_OFF))
    ap.add_argument("--test-per-bucket", type=int, default=4000)
    ap.add_argument("--train-rows", type=int, default=1_500_000)
    ap.add_argument("--off-test-frac", type=float, default=0.5)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "e5_results.json"))
    args = ap.parse_args()

    params = {
        "objective": "regression", "metric": "mae", "num_leaves": args.leaves,
        "learning_rate": args.learning_rate, "n_estimators": args.estimators,
        "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.9, "verbose": -1, "num_threads": args.threads,
    }

    rng = np.random.default_rng(args.seed)
    grid = pd.read_parquet(args.grid)
    off = pd.read_parquet(args.off)
    print(f"[e5] grid={len(grid):,}  off-network={len(off):,}")

    # Balanced grid test set (same construction as experiments.py).
    dist = grid["osrm_distance_m"].to_numpy(np.float64)
    labels = bucket_labels(None, dist)
    test_idx = []
    for name, _, _ in BUCKETS:
        idx = np.flatnonzero(labels == name)
        test_idx.append(rng.choice(idx, size=min(args.test_per_bucket, len(idx)), replace=False))
    test_idx = np.concatenate(test_idx)
    mask = np.ones(len(grid), dtype=bool)
    mask[test_idx] = False
    grid_train = grid.iloc[np.flatnonzero(mask)]
    if args.train_rows and len(grid_train) > args.train_rows:
        grid_train = grid_train.iloc[rng.choice(len(grid_train), size=args.train_rows, replace=False)]
    grid_test = grid.iloc[test_idx]

    # Split off-network rows into train-extra and test.
    perm = rng.permutation(len(off))
    n_off_test = int(len(off) * args.off_test_frac)
    off_test = off.iloc[perm[:n_off_test]]
    off_train = off.iloc[perm[n_off_test:]]
    print(f"[e5] grid_train={len(grid_train):,} grid_test={len(grid_test):,} "
          f"off_train={len(off_train):,} off_test={len(off_test):,}")

    results = {}
    t0 = time.time()
    a = train(grid_train, params, "inv_freq")
    results["A_grid_only__grid_test"] = evaluate(a, grid_test, "A grid-only -> grid test")
    results["A_grid_only__off_test"] = evaluate(a, off_test, "A grid-only -> off-network test")

    combined = pd.concat([grid_train, off_train], ignore_index=True)
    b = train(combined, params, "inv_freq")
    results["B_grid_plus_off__grid_test"] = evaluate(b, grid_test, "B grid+off -> grid test")
    results["B_grid_plus_off__off_test"] = evaluate(b, off_test, "B grid+off -> off-network test")
    print(f"[e5] total {time.time() - t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"params": params, "results": results}, indent=2))
    print(f"[e5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
