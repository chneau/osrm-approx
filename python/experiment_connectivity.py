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
"""Experiment 1 (connectivity/detour): the residual Experiment 0 left behind.

Experiment 0 gave the model the exact OSRM snap and it still could not approach
on-network quality: two points close in a straight line can be far apart by road
when a river, railway or motorway forces a detour. Those barriers are static, so
they can be revealed with O(1) raster lookups (build_connectivity_raster.py):

  * nearest-road class at each endpoint (local .. motorway);
  * distance from each endpoint to the nearest barrier;
  * how many water / rail / motorway barriers the straight segment crosses.

Feature sets (identical split, seeds vary training):

  raw               -- shipped 8.
  conn              -- raw + connectivity features.
  oracle_snap       -- Experiment 0's exact-snap features (the ceiling for snap).
  oracle_snap_conn  -- exact snap + connectivity (does connectivity beat the ceiling?).

    uv run experiment_connectivity.py --seeds 42 43 44
    uv run experiment_connectivity.py --quick          # one seed, raw+conn — does it move at all?
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
from experiments import BUCKETS, ROOT, bucket_labels, build_features, haversine, make_weights  # noqa: E402
from experiment_oracle_snap import SnapLookup, oracle_features  # noqa: E402

CONN_PATH = ROOT / "data" / "processed" / "connectivity.npz"
# Feature-set names are composed of these tokens.
USE_SNAP = {"oracle_snap", "oracle_snap_conn"}
USE_CONN = {"conn", "oracle_snap_conn"}


class ConnRaster:
    def __init__(self, path: Path, samples: int = 48):
        import json as _json
        d = np.load(path, allow_pickle=False)
        self.meta = _json.loads(str(d["meta"]))
        self.road_class = d["road_class"]
        self.water = d["water"]
        self.rail = d["rail"]
        self.motor = d["motor"]
        self.bar_dist = d["bar_dist"]
        self.samples = samples

    def _idx(self, lat, lon):
        m = self.meta
        ic = np.clip(np.floor((np.asarray(lon, np.float64) - m["min_lon"]) / m["cell_lon"]).astype(np.int64), 0, m["nx"] - 1)
        ir = np.clip(np.floor((np.asarray(lat, np.float64) - m["min_lat"]) / m["cell_lat"]).astype(np.int64), 0, m["ny"] - 1)
        return ir, ic

    def point(self, lat, lon):
        ir, ic = self._idx(lat, lon)
        return (self.road_class[ir, ic].astype(np.float64),
                self.bar_dist[ir, ic].astype(np.float64))

    def _sample(self, mask, la1, lo1, la2, lo2):
        """mask value at `samples` points along each segment -> [N, samples]."""
        n = len(la1)
        t = np.linspace(0.0, 1.0, self.samples)
        lat = la1[:, None] + (la2 - la1)[:, None] * t[None, :]
        lon = lo1[:, None] + (lo2 - lo1)[:, None] * t[None, :]
        ir, ic = self._idx(lat.ravel(), lon.ravel())
        return mask[ir, ic].reshape(n, self.samples)

    def crossings(self, la1, lo1, la2, lo2):
        """(#water runs, #rail runs, #motorway runs, total barrier cells) per pair."""
        sw = self._sample(self.water, la1, lo1, la2, lo2)
        sr = self._sample(self.rail, la1, lo1, la2, lo2)
        sm = self._sample(self.motor, la1, lo1, la2, lo2)
        runs = lambda s: (s[:, 1:] & ~s[:, :-1]).sum(axis=1).astype(np.float64)
        total = (sw.astype(np.int8) + sr.astype(np.int8) + sm.astype(np.int8)).sum(axis=1).astype(np.float64)
        return runs(sw), runs(sr), runs(sm), total

    def road_coverage(self, la1, lo1, la2, lo2):
        """(road fraction, longest non-road gap in metres, #non-road runs) on the segment.

        Two points genuinely connected locally have the straight line between them
        almost entirely on roads; a river/field/motorway strip in between shows up
        as a long non-road gap and is what forces the detour.
        """
        seg = haversine(la1, lo1, la2, lo2)
        s = self._sample(self.road_class > 0, la1, lo1, la2, lo2)  # (n, S) True=road
        frac = s.mean(axis=1)
        n, S = s.shape
        best = np.zeros(n, dtype=np.int64)
        cur = np.zeros(n, dtype=np.int64)
        for j in range(S):
            cur = np.where(s[:, j], 0, cur + 1)
            best = np.maximum(best, cur)
        gap_m = best * (seg / max(S - 1, 1))
        runs = (s[:, 1:] & ~s[:, :-1]).sum(axis=1).astype(np.float64)  # road -> non-road transitions
        return frac.astype(np.float64), gap_m.astype(np.float64), runs


def _conn_block(df, conn):
    la1 = df["orig_lat"].to_numpy(np.float64); lo1 = df["orig_lon"].to_numpy(np.float64)
    la2 = df["dest_lat"].to_numpy(np.float64); lo2 = df["dest_lon"].to_numpy(np.float64)
    cls_o, bd_o = conn.point(la1, lo1)
    cls_d, bd_d = conn.point(la2, lo2)
    w, r, m, tot = conn.crossings(la1, lo1, la2, lo2)
    road_frac, gap_m, gap_runs = conn.road_coverage(la1, lo1, la2, lo2)
    return np.column_stack([cls_o, cls_d, bd_o, bd_d, w, r, m, tot,
                            road_frac, gap_m, gap_runs]).astype(np.float32)


def build(df, kind, lookup, conn, chunk=200_000):
    if kind in USE_SNAP:
        X = oracle_features(df, "oracle_snap", lookup)
    else:
        X = build_features(df, "base")[0]
    if kind not in USE_CONN:
        return X
    extras = [_conn_block(df.iloc[i:i + chunk], conn) for i in range(0, len(df), chunk)]
    return np.column_stack([X, np.vstack(extras)])


def train(X, df, params, extra_w=None):
    w = make_weights(df["osrm_distance_m"].to_numpy(np.float64), "inv_freq")
    if extra_w is not None:
        w = w * extra_w
        w *= len(w) / w.sum()
    out = {}
    for key in ("osrm_distance_m", "osrm_duration_s"):
        y = df[key].to_numpy(np.float64)
        out[key] = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w).booster_
    return out


def ape_stats(truth, pred):
    ape = np.abs(truth - pred) / np.maximum(truth, 1e-6) * 100.0
    return {"n": int(len(truth)), "medape": float(np.median(ape)), "p90_ape": float(np.percentile(ape, 90)),
            "mae": float(np.mean(np.abs(truth - pred))), "medae": float(np.median(np.abs(truth - pred)))}


def evaluate(boosters, X, df, tag):
    res = {}
    print(f"[conn] {tag}  (n={len(df):,})")
    for key in ("osrm_distance_m", "osrm_duration_s"):
        truth = df[key].to_numpy(np.float64)
        pred = boosters[key].predict(X)
        labels = bucket_labels(None, truth)
        by = {name: ape_stats(truth[labels == name], pred[labels == name])
              for name, _, _ in BUCKETS if (labels == name).sum() >= 20}
        res[key] = {**ape_stats(truth, pred), "by_bucket": by}
        s = " ".join(f"{b}:{v['medape']:.0f}%" for b, v in by.items())
        print(f"[conn]   {key:16s} MedAPE={res[key]['medape']:5.1f}% MAE={res[key]['mae']:>9,.0f}  [{s}]")
    return res


def diagnostic(off, lookup, conn):
    """Model-independent: does road coverage on the straight segment explain the detour?"""
    la1 = off["orig_lat"].to_numpy(np.float64); lo1 = off["orig_lon"].to_numpy(np.float64)
    la2 = off["dest_lat"].to_numpy(np.float64); lo2 = off["dest_lon"].to_numpy(np.float64)
    s1lat, s1lon = lookup.snapped(la1, lo1)
    s2lat, s2lon = lookup.snapped(la2, lo2)
    sep = haversine(s1lat, s1lon, s2lat, s2lon)
    y = off["osrm_distance_m"].to_numpy(np.float64)
    road_frac, gap_m, gap_runs = conn.road_coverage(la1, lo1, la2, lo2)
    w, r, m, tot = conn.crossings(la1, lo1, la2, lo2)
    detour = y / np.maximum(sep, 1.0)
    print("[conn] diagnostic: off-network-short detour factor (osrm / snapped separation)")
    print(f"[conn]   {'road frac':>10s} {'n':>6s} {'med detour':>11s} {'med gap m':>10s}")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (road_frac >= lo) & (road_frac < hi)
        if msk.sum() < 20:
            continue
        print(f"[conn]   {lo:.1f}-{hi:.1f}   {int(msk.sum()):>6d} {np.median(detour[msk]):>11.2f} {np.median(gap_m[msk]):>10.0f}")
    print(f"[conn]   {'barrier runs':>10s} {'n':>6s} {'med detour':>11s}")
    runs = (w + r + m).astype(int)
    for k in range(0, 5):
        msk = runs == k if k < 4 else runs >= 4
        if msk.sum() < 20:
            continue
        print(f"[conn]   {'%d' % k if k < 4 else '4+':>10s}   {int(msk.sum()):>6d} {np.median(detour[msk]):>11.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off", default=str(ROOT / "data" / "processed" / "offnetwork_short.parquet"))
    ap.add_argument("--grid", default=str(ROOT / "data" / "processed" / "samples.parquet"))
    ap.add_argument("--snap", default=str(ROOT / "data" / "processed" / "snap_endpoints.parquet"))
    ap.add_argument("--conn", default=str(CONN_PATH))
    ap.add_argument("--kinds", nargs="+", default=["raw", "conn", "oracle_snap", "oracle_snap_conn"])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--grid-test-per-bucket", type=int, default=4000)
    ap.add_argument("--train-rows", type=int, default=2_000_000)
    ap.add_argument("--off-test-frac", type=float, default=0.5)
    ap.add_argument("--leaves", type=int, default=511)
    ap.add_argument("--estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="iteration shortcut: one seed, raw+conn only (does the feature even move?)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "connectivity_results.json"))
    args = ap.parse_args()
    if args.quick:
        args.seeds = [args.seeds[0]]
        args.kinds = ["raw", "conn"]

    lookup = SnapLookup(Path(args.snap))
    conn = ConnRaster(Path(args.conn))
    print(f"[conn] connectivity raster samples/segment = {conn.samples}")

    rng = np.random.default_rng(args.split_seed)
    grid = pd.read_parquet(args.grid)
    off = pd.read_parquet(args.off)
    print(f"[conn] grid={len(grid):,} off-network-short={len(off):,}")

    dist = grid["osrm_distance_m"].to_numpy(np.float64)
    labels = bucket_labels(None, dist)
    test_idx = []
    for name, _, _ in BUCKETS:
        idx = np.flatnonzero(labels == name)
        test_idx.append(rng.choice(idx, size=min(args.grid_test_per_bucket, len(idx)), replace=False))
    test_idx = np.concatenate(test_idx)
    keep = np.ones(len(grid), dtype=bool); keep[test_idx] = False
    grid_train = grid.iloc[np.flatnonzero(keep)]
    if args.train_rows and len(grid_train) > args.train_rows:
        grid_train = grid_train.iloc[rng.choice(len(grid_train), size=args.train_rows, replace=False)]
    grid_test = grid.iloc[test_idx].reset_index(drop=True)

    perm = rng.permutation(len(off))
    n_off_test = int(len(off) * args.off_test_frac)
    off_test = off.iloc[perm[:n_off_test]].reset_index(drop=True)
    off_train = off.iloc[perm[n_off_test:]].reset_index(drop=True)
    print(f"[conn] grid_train={len(grid_train):,} grid_test={len(grid_test):,} off_train={len(off_train):,} off_test={len(off_test):,}")

    if "conn" in args.kinds or any(k in USE_CONN for k in args.kinds):
        diagnostic(off, lookup, conn)

    combined = pd.concat([grid_train, off_train], ignore_index=True)
    results = {"split_seed": args.split_seed, "params": {"leaves": args.leaves, "estimators": args.estimators,
               "learning_rate": args.learning_rate, "train_rows": args.train_rows}, "runs": []}
    t0 = time.time()
    # Kind-major so each (kind, dataset) feature matrix is built once and reused
    # across every seed. The split is fixed, so X does not depend on the training
    # seed; building it inside the seed loop recomputed it 3x for nothing.
    for kind in args.kinds:
        tb = time.time()
        X_train = build(combined, kind, lookup, conn)
        X_off = build(off_test, kind, lookup, conn)
        X_grid = build(grid_test, kind, lookup, conn)
        print(f"[conn] built {kind} features in {time.time() - tb:.0f}s "
              f"(train {X_train.shape}, off {X_off.shape}, grid {X_grid.shape})")
        for seed in args.seeds:
            params = {"objective": "regression", "metric": "mae", "num_leaves": args.leaves,
                      "learning_rate": args.learning_rate, "n_estimators": args.estimators,
                      "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
                      "colsample_bytree": 0.9, "verbose": -1, "num_threads": args.threads, "random_state": seed}
            print(f"\n[conn] === seed={seed} features={kind} ===")
            boosters = train(X_train, combined, params)
            run = {"seed": seed, "kind": kind}
            run["off_test"] = evaluate(boosters, X_off, off_test, f"seed{seed} {kind} -> off-short")
            run["grid_test"] = evaluate(boosters, X_grid, grid_test, f"seed{seed} {kind} -> grid")
            results["runs"].append(run)
    results["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[conn] total {results['wall_s']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
