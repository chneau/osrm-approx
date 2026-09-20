#!/usr/bin/env python
"""S1 prototype: ship the all-pairs matrix, interpolate. Plus fixed-point sizing.

Uses the OSRM truth already in data/processed/samples.parquet (the 2,203 routable
grid nodes) as the model. A query (A,B) is answered by finding the k nearest grid
nodes to each endpoint and taking the separable weighted average of the matrix.

Compares against the shipped ONNX tree ensemble on the same held-out raw-coordinate
pairs, so the headline numbers are like-for-like.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

BUCKETS = [("<1km", 0, 1_000), ("1-3km", 1_000, 3_000), ("3-10km", 3_000, 10_000),
           (">25km", 25_000, np.inf)]


def metrics(y, p):
    e = np.abs(y - p)
    rel = np.where(y > 1e-6, e / y, np.nan)
    return dict(n=len(y), mae=float(e.mean()), medae=float(np.median(e)),
                medape=float(np.nanmedian(rel) * 100), within10=float((rel <= 0.10).mean() * 100))


def geometric_features(o_lat, o_lon, d_lat, d_lon):
    R = 6_371_008.8
    p1, p2 = np.radians(o_lat), np.radians(d_lat)
    dp, dl = p2 - p1, np.radians(d_lon - o_lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    hav = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    brg = np.degrees(np.arctan2(y, x)) % 360
    return np.column_stack([o_lat, o_lon, d_lat, d_lon, hav, brg, d_lat - o_lat, d_lon - o_lon]).astype(np.float32)


def main():
    t0 = time.time()
    samples = pd.read_parquet(PROC / "samples.parquet")
    # Nodes = unique endpoints of the all-pairs grid, via exact fixed-point keys so
    # float32 round-tripping cannot drop a coordinate.
    pts = np.concatenate([
        samples[["orig_lat", "orig_lon"]].to_numpy(np.float64),
        samples[["dest_lat", "dest_lon"]].to_numpy(np.float64),
    ])
    nodes, inv = np.unique(pts, axis=0, return_inverse=True)
    i = inv[: len(samples)].astype(np.int32)
    j = inv[len(samples):].astype(np.int32)
    N = len(nodes)
    print(f"[s1] nodes={N}  samples={len(samples):,}  ({time.time()-t0:.1f}s)")

    D = np.full((N, N), np.nan, np.float32)
    T = np.full((N, N), np.nan, np.float32)
    D[i, j] = samples["osrm_distance_m"].to_numpy(np.float32)
    T[i, j] = samples["osrm_duration_s"].to_numpy(np.float32)
    n_missing = int(np.isnan(D).sum())
    # Fill the reverse pair where OSRM only returned one direction (one-ways).
    D = np.where(np.isnan(D), D.T, D)
    T = np.where(np.isnan(T), T.T, T)
    D = np.where(np.isnan(D), 0.0, D)
    T = np.where(np.isnan(T), 0.0, T)
    del samples, pts
    print(f"[s1] matrices filled ({time.time()-t0:.1f}s), missing cells={n_missing:,} "
          f"({n_missing / N / N * 100:.2f}%), residual NaNs={int(np.isnan(D).sum())}")

    # ---- evaluation sets: raw coordinates (the service's real input) ----
    off = pd.read_parquet(PROC / "offnetwork.parquet")
    rng = np.random.default_rng(0)
    m = rng.permutation(len(off))
    test = off.iloc[m[:40_000]].reset_index(drop=True)   # unseen raw coords
    print(f"[s1] test pairs from offnetwork.parquet: {len(test):,}")

    node_ll = nodes
    # Anisotropic scaling: 1 deg lon ~ cos(lat) deg lat at Manchester.
    scale = np.array([1.0, np.cos(np.radians(53.5))])

    def predict(k, power):
        nbr = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", metric="euclidean").fit(node_ll * scale)
        def one(lat, lon):
            d, idx = nbr.kneighbors(np.column_stack([lat, lon]) * scale)
            w = 1.0 / np.maximum(d, 1e-9) ** power
            return idx, w
        oi, ow = one(test["orig_lat"].to_numpy(), test["orig_lon"].to_numpy())
        di, dw = one(test["dest_lat"].to_numpy(), test["dest_lon"].to_numpy())
        # separable weighted average over the k x k matrix block per pair
        num_d = np.zeros(len(test)); num_t = np.zeros(len(test))
        for a in range(k):
            for b in range(k):
                ww = ow[:, a] * dw[:, b]
                num_d += ww * D[oi[:, a], di[:, b]]
                num_t += ww * T[oi[:, a], di[:, b]]
        den = ow.sum(1) * dw.sum(1)
        return num_d / den, num_t / den

    # ---- shipped ONNX baseline on the same rows ----
    import onnxruntime as ort
    X = geometric_features(test["orig_lat"].to_numpy(np.float64), test["orig_lon"].to_numpy(np.float64),
                           test["dest_lat"].to_numpy(np.float64), test["dest_lon"].to_numpy(np.float64))
    sess = ort.InferenceSession(str(ROOT / "server" / "models" / "model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(["distance_m", "duration_s"], {"features": X})
    onnx_d, onnx_t = got[0].ravel(), got[1].ravel()

    truth_d = test["osrm_distance_m"].to_numpy()
    truth_t = test["osrm_duration_s"].to_numpy()

    rows = []
    for name, (pd_, pt) in [("onnx-511", (onnx_d, onnx_t))]:
        for tgt, y, p in [("distance", truth_d, pd_), ("duration", truth_t, pt)]:
            rows.append((name, tgt, "ALL", metrics(y, p)))
    for k in (1, 4, 8):
        pd_, pt = predict(k, 2.0)
        for tgt, y, p in [("distance", truth_d, pd_), ("duration", truth_t, pt)]:
            rows.append((f"s1-k{k}", tgt, "ALL", metrics(y, p)))
    print("\n%-10s %-9s %-7s %8s %9s %8s %9s" % ("model", "target", "band", "n", "MedAPE%", "MAE", "within10%"))
    for name, tgt, band, m in rows:
        print("%-10s %-9s %-7s %8d %9.1f %8.1f %8.1f%%" % (name, tgt, band, m["n"], m["medape"], m["mae"], m["within10"]))

    # ---- fixed-point sizing ----
    print("\n[s1] fixed-point coordinate encodings (Greater Manchester bbox, offset from origin):")
    bbox_lat0, bbox_lon0 = 53.315, -2.760
    for dec, name in [(4, "1e-4 (~11 m)"), (5, "1e-5 (~1.1 m)"), (6, "1e-6 (~11 cm)"), (7, "1e-7 (~1.1 cm)")]:
        s = 10 ** dec
        la = int((53.665 - bbox_lat0) * s)
        lo = int((-1.890 - bbox_lon0) * s)
        bits = max(la, lo).bit_length()
        # signed 24-bit max = 2^23-1
        fit = "OK" if la <= 2**23 - 1 and lo <= 2**23 - 1 else "OVERFLOW signed24"
        print(f"  scale {name:>16}: lat_span={la:>10} lon_span={lo:>10} -> {bits} bits ({fit})")
    print("  (global lat/lon at 1e-5 needs 25 bits signed; 1e-4 needs 21 bits -> fits int24)")
    print("\n[s1] matrix sizing for N=%d, 2 targets:" % N)
    for label, nbytes in [("float32", 4), ("int32 (m, s)", 4), ("uint16 (m, s)", 2), ("int16 (decametre, second)", 2)]:
        print(f"  {label:>32}: {2 * N * N * nbytes / 1e6:8.1f} MB")

    print(f"\n[s1] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
