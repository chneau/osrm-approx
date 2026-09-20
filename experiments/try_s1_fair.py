#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "lightgbm>=4.7.0",
#   "numpy>=2.5.3",
#   "onnx>=1.23.0",
#   "onnxruntime>=1.30.0",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "scikit-learn>=1.9.1",
# ]
# ///
"""Fair S1 vs GBM bake-off on raw (off-network) coordinates.

The shipped ONNX model was trained on all of data/processed/offnetwork.parquet, so
evaluating it there is contaminated. Here we hold out 20% of those pairs and retrain
a GBM (same params as shipped) on samples.parquet + the other 80%, then score:
  - honest GBM (never saw the test pairs)
  - S1 k-NN over the 2,203 grid nodes (never saw off-network data at all)
  - S1-oracle: test endpoints injected as nodes (upper bound / cheating)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from train_export_onnx import geometric_features, inverse_frequency_weights  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

PROC = ROOT / "data" / "processed"
BUCKETS = [("<1km", 0, 1_000), ("1-3km", 1_000, 3_000), ("3-10km", 3_000, 10_000), (">25km", 25_000, np.inf)]


def metrics(y, p):
    e = np.abs(y - p); rel = np.where(y > 1e-6, e / y, np.nan)
    return dict(n=len(y), mae=float(e.mean()), medae=float(np.median(e)), medape=float(np.nanmedian(rel) * 100))


def report(tag, truth_d, pred_d, truth_t, pred_t):
    for name, y, p in (("distance", truth_d, pred_d), ("duration", truth_t, pred_t)):
        m = metrics(y, p)
        print(f"  {tag:12s} {name:9s} ALL      MedAPE={m['medape']:6.1f}%  MAE={m['mae']:9.1f}")
        for label, lo, hi in BUCKETS:
            mask = (y >= lo) & (y < hi)
            if mask.sum() < 100:
                continue
            mm = metrics(y[mask], p[mask])
            print(f"  {tag:12s} {name:9s} {label:7s}  MedAPE={mm['medape']:6.1f}%  n={mm['n']}")


def main():
    t0 = time.time()
    samples = pd.read_parquet(PROC / "samples.parquet")
    off = pd.read_parquet(PROC / "offnetwork.parquet")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(off))
    te = off.iloc[perm[: len(off) // 5]].reset_index(drop=True)
    tr_off = off.iloc[perm[len(off) // 5:]].reset_index(drop=True)
    print(f"[fair] samples={len(samples):,}  off-train={len(tr_off):,}  off-test={len(te):,}")

    # ---- S1 over grid nodes only ----
    def build_matrix(df):
        pts = np.concatenate([df[["orig_lat", "orig_lon"]].to_numpy(np.float64),
                              df[["dest_lat", "dest_lon"]].to_numpy(np.float64)])
        nodes, inv = np.unique(pts, axis=0, return_inverse=True)
        n = len(nodes); i, j = inv[:len(df)], inv[len(df):]
        D = np.full((n, n), np.nan, np.float32); T = np.full((n, n), np.nan, np.float32)
        D[i, j] = df["osrm_distance_m"].to_numpy(np.float32); T[i, j] = df["osrm_duration_s"].to_numpy(np.float32)
        D = np.where(np.isnan(D), D.T, D); T = np.where(np.isnan(T), T.T, T)
        return nodes, np.nan_to_num(D), np.nan_to_num(T)

    nodes, D, T = build_matrix(samples)
    scale = np.array([1.0, np.cos(np.radians(53.5))])
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(nodes * scale)
    oi = nn.kneighbors(te[["orig_lat", "orig_lon"]].to_numpy() * scale, return_distance=False).ravel()
    di = nn.kneighbors(te[["dest_lat", "dest_lon"]].to_numpy() * scale, return_distance=False).ravel()
    s1_d, s1_t = D[oi, di], T[oi, di]
    print(f"[fair] S1 grid lookup built ({time.time()-t0:.1f}s)")

    truth_d = te["osrm_distance_m"].to_numpy(); truth_t = te["osrm_duration_s"].to_numpy()

    # ---- honest GBM trained on samples + remaining off-network ----
    print("[fair] training GBM (511 leaves, 400 trees, inv-freq weights)...")
    both = pd.concat([samples, tr_off], ignore_index=True)
    X = geometric_features(both["orig_lat"].to_numpy(), both["orig_lon"].to_numpy(),
                           both["dest_lat"].to_numpy(), both["dest_lon"].to_numpy())
    w = inverse_frequency_weights(both["osrm_distance_m"].to_numpy(np.float64))
    Xte = geometric_features(te["orig_lat"].to_numpy(), te["orig_lon"].to_numpy(),
                             te["dest_lat"].to_numpy(), te["dest_lon"].to_numpy())
    params = dict(objective="regression", metric="mae", num_leaves=511, learning_rate=0.08,
                  n_estimators=400, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.9, verbose=-1, num_threads=16)
    preds = {}
    for tgt, col in (("distance", "osrm_distance_m"), ("duration", "osrm_duration_s")):
        m = lgb.LGBMRegressor(**params).fit(X, both[col].to_numpy(np.float32), sample_weight=w)
        preds[tgt] = m.predict(Xte)
        print(f"[fair]   {tgt} trained ({time.time()-t0:.0f}s)")

    print("\n================ HONEST COMPARISON (raw off-network coords) ================")
    report("GBM-honest", truth_d, preds["distance"], truth_t, preds["duration"])
    report("S1-grid", truth_d, s1_d, truth_t, s1_t)
    print(f"\n[fair] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
