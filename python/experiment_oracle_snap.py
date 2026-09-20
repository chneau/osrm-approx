#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "lightgbm>=4.7.0",
#   "numpy>=2.5.3",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "scikit-learn>=1.9.1",
# ]
# ///
"""Experiment 0 (oracle snap): is snapping the short-trip bottleneck?

The service is handed raw coordinates, but OSRM routes between the *snapped*
points. E13 tried a coarse 111 m raster to reveal that snap and got a real but
modest win. Experiment 0 replaces the raster with the **exact** snap OSRM uses
(`snap_endpoints.py`, one `/nearest` per endpoint, build-time). That is the
ceiling any snapper could reach, so it answers two things at once:

  * if the oracle closes the off-network short-trip gap -> snapping is the story
    and the work is to build a snapper; the raster-vs-oracle gap is the fidelity
    budget.
  * if it barely moves -> the remaining error is connectivity/detour, not snap.

Two feature sets on the identical split (fixed test set, seeds vary training):

  raw          -- the current 8 base features (ship today).
  oracle_snap  -- base + exact snap displacement (east/north) at both ends +
                  snapped-point separation & bearing (16 features).

    uv run experiment_oracle_snap.py --seeds 42 43 44
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
    bearing,
    bucket_labels,
    build_features,
    haversine,
    make_weights,
)


class SnapLookup:
    """Exact OSRM snap per endpoint (the coordinate OSRM routes from), keyed by
    rounded (lat, lon). Displacement is derived from the exact snapped coordinate
    via haversine+bearing -- not a flat-earth projection, which drifts ~100 m on
    the far-snapping points (up to 18 km away here)."""

    def __init__(self, path: Path):
        d = pd.read_parquet(path)
        key = list(zip(np.round(d["lat"].to_numpy(np.float64), 6),
                       np.round(d["lon"].to_numpy(np.float64), 6)))
        self.slat = dict(zip(key, d["snap_lat"].to_numpy(np.float64)))
        self.slon = dict(zip(key, d["snap_lon"].to_numpy(np.float64)))

    def snapped(self, lat, lon):
        key = list(zip(np.round(np.asarray(lat, np.float64), 6),
                       np.round(np.asarray(lon, np.float64), 6)))
        slat = np.array([self.slat.get(k, np.nan) for k in key], np.float64)
        slon = np.array([self.slon.get(k, np.nan) for k in key], np.float64)
        return slat, slon


def snap_displacement(lat, lon, slat, slon):
    """East/north component (m) and magnitude of raw -> snapped."""
    d = haversine(lat, lon, slat, slon)
    b = np.radians(bearing(lat, lon, slat, slon))
    return d * np.sin(b), d * np.cos(b), d


def oracle_features(df, kind, lookup):
    base, hav = build_features(df, "base")
    if kind == "raw":
        return base
    if kind != "oracle_snap":
        raise ValueError(kind)
    lat1 = df["orig_lat"].to_numpy(np.float64)
    lon1 = df["orig_lon"].to_numpy(np.float64)
    lat2 = df["dest_lat"].to_numpy(np.float64)
    lon2 = df["dest_lon"].to_numpy(np.float64)
    slat1, slon1 = lookup.snapped(lat1, lon1)
    slat2, slon2 = lookup.snapped(lat2, lon2)
    if np.isnan(slat1).any() or np.isnan(slat2).any():
        raise RuntimeError("snap lookup missed endpoints")
    oe, on, od = snap_displacement(lat1, lon1, slat1, slon1)
    de, dn, dd = snap_displacement(lat2, lon2, slat2, slon2)
    snap_hav = haversine(slat1, slon1, slat2, slon2)
    snap_brg = bearing(slat1, slon1, slat2, slon2)
    return np.column_stack([base, od, oe, on, dd, de, dn, snap_hav, snap_brg]).astype(np.float32)


def train(X, df, params, extra_w=None):
    w = make_weights(df["osrm_distance_m"].to_numpy(np.float64), "inv_freq")
    if extra_w is not None:
        w = w * extra_w
        w *= len(w) / w.sum()
    boosters = {}
    for key in ("osrm_distance_m", "osrm_duration_s"):
        y = df[key].to_numpy(np.float64)
        boosters[key] = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w).booster_
    return boosters


def ape_stats(truth, pred):
    ape = np.abs(truth - pred) / np.maximum(truth, 1e-6) * 100.0
    return {
        "n": int(len(truth)),
        "medape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
        "mae": float(np.mean(np.abs(truth - pred))),
        "medae": float(np.median(np.abs(truth - pred))),
    }


def evaluate(boosters, X, df, tag):
    out = {}
    print(f"[oracle] {tag}  (n={len(df):,})")
    for key in ("osrm_distance_m", "osrm_duration_s"):
        truth = df[key].to_numpy(np.float64)
        pred = boosters[key].predict(X)
        labels = bucket_labels(None, truth)
        by = {}
        for name, _, _ in BUCKETS:
            m = labels == name
            if m.sum() < 20:
                continue
            by[name] = ape_stats(truth[m], pred[m])
        allst = ape_stats(truth, pred)
        out[key] = {**allst, "by_bucket": by}
        buckets = " ".join(f"{b}:{v['medape']:.0f}%(p90 {v['p90_ape']:.0f}%)" for b, v in by.items())
        print(f"[oracle]   {key:16s} MedAPE={allst['medape']:5.1f}% p90APE={allst['p90_ape']:6.1f}% "
              f"MAE={allst['mae']:>9,.0f}")
        print(f"[oracle]     by bucket: {buckets}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off", default=str(ROOT / "data" / "processed" / "offnetwork_short.parquet"))
    ap.add_argument("--grid", default=str(ROOT / "data" / "processed" / "samples.parquet"))
    ap.add_argument("--snap", default=str(ROOT / "data" / "processed" / "snap_endpoints.parquet"))
    ap.add_argument("--kinds", nargs="+", default=["raw", "oracle_snap"])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--grid-test-per-bucket", type=int, default=4000)
    ap.add_argument("--train-rows", type=int, default=2_000_000)
    ap.add_argument("--off-test-frac", type=float, default=0.5)
    ap.add_argument("--leaves", type=int, default=511)
    ap.add_argument("--estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--off-boost", type=float, default=1.0,
                    help="extra sample weight on the off-network rows (they are ~0.1%% of the pool)")
    ap.add_argument("--quick", action="store_true", help="iteration shortcut: one seed only")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "oracle_snap_results.json"))
    args = ap.parse_args()
    if args.quick:
        args.seeds = [args.seeds[0]]

    lookup = SnapLookup(Path(args.snap))

    rng = np.random.default_rng(args.split_seed)
    grid = pd.read_parquet(args.grid)
    off = pd.read_parquet(args.off)
    print(f"[oracle] grid={len(grid):,}  off-network-short={len(off):,}")

    # Fixed balanced grid test set; the grid train pool is the rest.
    dist = grid["osrm_distance_m"].to_numpy(np.float64)
    labels = bucket_labels(None, dist)
    test_idx = []
    for name, _, _ in BUCKETS:
        idx = np.flatnonzero(labels == name)
        test_idx.append(rng.choice(idx, size=min(args.grid_test_per_bucket, len(idx)), replace=False))
    test_idx = np.concatenate(test_idx)
    keep = np.ones(len(grid), dtype=bool)
    keep[test_idx] = False
    grid_train = grid.iloc[np.flatnonzero(keep)]
    if args.train_rows and len(grid_train) > args.train_rows:
        grid_train = grid_train.iloc[rng.choice(len(grid_train), size=args.train_rows, replace=False)]
    grid_test = grid.iloc[test_idx].reset_index(drop=True)

    # Fixed off-network split.
    perm = rng.permutation(len(off))
    n_off_test = int(len(off) * args.off_test_frac)
    off_test = off.iloc[perm[:n_off_test]].reset_index(drop=True)
    off_train = off.iloc[perm[n_off_test:]].reset_index(drop=True)
    print(f"[oracle] grid_train={len(grid_train):,} grid_test={len(grid_test):,} "
          f"off_train={len(off_train):,} off_test={len(off_test):,}")

    combined = pd.concat([grid_train, off_train], ignore_index=True)
    extra_w = None
    if args.off_boost != 1.0:
        extra_w = np.ones(len(combined))
        extra_w[len(grid_train):] = args.off_boost

    results = {"split_seed": args.split_seed, "off_boost": args.off_boost, "params": {
        "leaves": args.leaves, "estimators": args.estimators,
        "learning_rate": args.learning_rate, "train_rows": args.train_rows},
        "runs": []}
    t0 = time.time()
    # Kind-major so each (kind, dataset) feature matrix is built once and reused
    # across every seed (the split is fixed, so X is seed-independent).
    for kind in args.kinds:
        tb = time.time()
        X_train = oracle_features(combined, kind, lookup)
        X_off = oracle_features(off_test, kind, lookup)
        X_grid = oracle_features(grid_test, kind, lookup)
        print(f"[oracle] built {kind} features in {time.time() - tb:.0f}s "
              f"(train {X_train.shape}, off {X_off.shape}, grid {X_grid.shape})")
        for seed in args.seeds:
            params = {
                "objective": "regression", "metric": "mae", "num_leaves": args.leaves,
                "learning_rate": args.learning_rate, "n_estimators": args.estimators,
                "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.9, "verbose": -1, "num_threads": args.threads,
                "random_state": seed,
            }
            print(f"\n[oracle] === seed={seed} features={kind} ===")
            boosters = train(X_train, combined, params, extra_w)
            run = {"seed": seed, "kind": kind}
            run["off_test"] = evaluate(boosters, X_off, off_test, f"seed{seed} {kind} -> off-short test")
            run["grid_test"] = evaluate(boosters, X_grid, grid_test, f"seed{seed} {kind} -> grid test")
            results["runs"].append(run)
    results["wall_s"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[oracle] total {results['wall_s']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
