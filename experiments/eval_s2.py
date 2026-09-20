#!/usr/bin/env python
"""S2 step 2: validate the hub-label oracle and score it against S1/GBM.

- Correctness: random node pairs, PLL query vs scipy shortest path on the same graph.
- Raw coords: snap to the nearest S2 node, query PLL, compare to OSRM truth on the
  same 20% off-network split used by experiments/try_s1_fair.py.
- Latency: per-query time for a batch of PLL lookups.
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BUCKETS = [("<1km", 0, 1_000), ("1-3km", 1_000, 3_000), ("3-10km", 3_000, 10_000), (">25km", 25_000, np.inf)]


def load_labels(path):
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        out = []
        for _ in range(n):
            c = struct.unpack("<i", f.read(4))[0]
            arr = np.frombuffer(f.read(8 * c), dtype=np.dtype([("hub", "<i4"), ("dist", "<f4")]))
            out.append(dict(zip(arr["hub"].tolist(), arr["dist"].tolist())))
    return out


def q(la, lb):
    if len(lb) < len(la):
        la, lb = lb, la
    best = float("inf")
    get = lb.get
    for h, dv in la.items():
        du = get(h)
        if du is not None:
            s = dv + du
            if s < best:
                best = s
    return best


def metrics(y, p):
    e = np.abs(y - p); rel = np.where(y > 1e-6, e / y, np.nan)
    return dict(n=len(y), medape=float(np.nanmedian(rel) * 100), mae=float(e.mean()))


def main():
    t0 = time.time()
    g = np.load(ROOT / "experiments" / "s2_graph.npz")
    n = int(g["n"]); lat = g["lat"]; lon = g["lon"]; u = g["u"]; v = g["v"]
    d = g["dist"].astype(np.float64)
    print(f"[s2] graph n={n} m={len(u)}")
    labels = load_labels(ROOT / "experiments" / "s2_dist.labels")
    lset = load_labels(ROOT / "experiments" / "s2_dur.labels")
    print(f"[s2] labels loaded (avg {np.mean([len(x) for x in labels]):.0f}/node)  ({time.time()-t0:.1f}s)")

    # ---------- correctness vs scipy ----------
    rows = np.concatenate([u, v]); cols = np.concatenate([v, u]); w = np.concatenate([d, d])
    G = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    rng = np.random.default_rng(1)
    src = rng.choice(n, 120, replace=False)
    Dtrue = dijkstra(G, directed=False, indices=src)
    r = rng.integers(0, n, size=(4000, 2))
    pll = np.array([q(labels[a], labels[b]) for a, b in r])
    # compare PLL to scipy for pairs whose source is one of the sampled sources
    src_list = list(src)
    sel = np.isin(r[:, 0], src)
    true = np.array([Dtrue[src_list.index(a), b] for a, b in r[sel]])
    err = np.abs(pll[sel] - true)
    print(f"[s2] correctness vs scipy on {sel.sum()} pairs: max|Δ|={err.max():.4f} m, "
          f"mean={err.mean():.6f}, exact<1e-3: {(err < 1e-3).mean()*100:.2f}%")

    # ---------- latency ----------
    qpairs = rng.integers(0, n, size=(20000, 2))
    t = time.time()
    for a, b in qpairs:
        q(labels[a], labels[b])
    print(f"[s2] PLL query latency: {(time.time()-t)/len(qpairs)*1e6:.1f} us/query (python dict)")

    # ---------- raw-coordinate accuracy (same split as try_s1_fair) ----------
    off = pd.read_parquet(PROC / "offnetwork.parquet")
    rr = np.random.default_rng(0); perm = rr.permutation(len(off))
    te = off.iloc[perm[: len(off) // 5]].reset_index(drop=True)
    node_ll = np.column_stack([lat, lon]); scale = np.array([1.0, np.cos(np.radians(53.5))])
    nn = NearestNeighbors(n_neighbors=1).fit(node_ll * scale)
    oi = nn.kneighbors(te[["orig_lat", "orig_lon"]].to_numpy() * scale, return_distance=False).ravel()
    di = nn.kneighbors(te[["dest_lat", "dest_lon"]].to_numpy() * scale, return_distance=False).ravel()
    print(f"[s2] snapping test pairs to graph nodes ({time.time()-t0:.1f}s)")

    t = time.time()
    s2_d = np.array([q(labels[a], labels[b]) for a, b in zip(oi, di)])
    s2_t = np.array([q(lset[a], lset[b]) for a, b in zip(oi, di)])
    print(f"[s2] {len(te):,} PLL queries in {time.time()-t:.2f}s")

    truth_d = te["osrm_distance_m"].to_numpy(); truth_t = te["osrm_duration_s"].to_numpy()
    for tag, y, p in (("S2-distance", truth_d, s2_d), ("S2-duration", truth_t, s2_t)):
        m = metrics(y, p)
        print(f"\n  {tag:12s} ALL      MedAPE={m['medape']:6.1f}%  MAE={m['mae']:9.1f}")
        for label, lo, hi in BUCKETS:
            mask = (y >= lo) & (y < hi)
            if mask.sum() < 100:
                continue
            mm = metrics(y[mask], p[mask])
            print(f"  {tag:12s} {label:7s}  MedAPE={mm['medape']:6.1f}%  n={mm['n']}")
    print(f"\n[s2] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
