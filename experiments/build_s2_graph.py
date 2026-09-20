#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy>=2.5.3",
#   "requests>=2.34.2",
#   "scikit-learn>=1.9.1",
# ]
# ///
"""S2 step 1: build a sparse road graph at finer-than-training resolution.

Nodes = a multi-resolution grid snapped to OSRM (snap <= max_snap_m). Edges = each
node to its k nearest other nodes, weighted with the *exact* OSRM /table distance and
duration between them. This keeps OSRM's routing as ground truth; the only error the
oracle can incur is grid discretisation.

Outputs experiments/s2_graph.npz (n, edges u/v/dist/dur) and a C-friendly binary
experiments/s2_graph.bin.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from generate_grid import build_grid  # noqa: E402

DEG = 111_320.0


def coord_str(lat, lon):
    return ";".join(f"{b:.6f},{a:.6f}" for a, b in zip(lat, lon))


def snap_all(lat, lon, url, workers=32):
    sess_local = {}
    def one(i):
        s = sess_local.get("s")
        if s is None:
            s = requests.Session(); sess_local["s"] = s
        r = s.get(f"{url}/nearest/v1/driving/{lon[i]:.6f},{lat[i]:.6f}?number=1", timeout=60)
        r.raise_for_status()
        w = r.json()["waypoints"][0]
        return w["distance"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.array(list(pool.map(one, range(len(lat)))))


def table(url, sources, dests, sess):
    """sources, dests: (lat,lon) arrays. Returns (dur[n_s,n_d], dist[n_s,n_d])."""
    n_s, n_d = len(sources[0]), len(dests[0])
    coords = coord_str(np.concatenate([sources[0], dests[0]]), np.concatenate([sources[1], dests[1]]))
    src = ";".join(str(i) for i in range(n_s))
    dst = ";".join(str(i) for i in range(n_s, n_s + n_d))
    u = f"{url}/table/v1/driving/{coords}?sources={src}&destinations={dst}&annotations=duration,distance"
    r = sess.get(u, timeout=300); r.raise_for_status()
    d = r.json()
    if d.get("code") != "Ok":
        raise RuntimeError(d)
    return np.asarray(d["durations"], np.float64), np.asarray(d["distances"], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--rings", default="5:0.002,15:0.004,40:0.008")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-snap-m", type=float, default=150.0)
    ap.add_argument("--chunk", type=int, default=400, help="sources per /table request")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "s2_graph.npz"))
    args = ap.parse_args()

    t0 = time.time()
    rings = [(float(a), float(b)) for a, b in (p.split(":") for p in args.rings.split(","))]
    g = build_grid(rings)
    print(f"[s2] raw grid points: {len(g)}  ({time.time()-t0:.1f}s)")
    lat = g["lat"].to_numpy(np.float64); lon = g["lon"].to_numpy(np.float64)

    snap = snap_all(lat, lon, args.url)
    keep = snap <= args.max_snap_m
    lat, lon = lat[keep], lon[keep]
    n = len(lat)
    print(f"[s2] routable nodes: {n} (dropped {int((~keep).sum())} off-network; snap p95={np.percentile(snap[keep],95):.0f}m)  ({time.time()-t0:.1f}s)")

    # kNN by great-circle-ish scaled degrees
    scale = np.array([1.0, np.cos(np.radians(53.5))])
    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(np.column_stack([lat, lon]) * scale)
    _, nbr = nn.kneighbors(np.column_stack([lat, lon]) * scale)
    nbr = nbr[:, 1:]  # drop self

    # one /table pass: sources = nodes, destinations = their k neighbours
    sess = requests.Session()
    edges_u, edges_v, edges_d, edges_t = [], [], [], []
    for s0 in range(0, n, args.chunk):
        s1 = min(s0 + args.chunk, n)
        src_lat, src_lon = lat[s0:s1], lon[s0:s1]
        dest_idx = nbr[s0:s1].reshape(-1)
        dur, dist = table(args.url, (src_lat, src_lon), (lat[dest_idx], lon[dest_idx]), sess)
        for r in range(s1 - s0):
            for c in range(args.k):
                col = r * args.k + c          # this source's own knn columns
                d = dist[r, col]
                if np.isfinite(d) and d > 0:
                    edges_u.append(s0 + r); edges_v.append(int(dest_idx[col]))
                    edges_d.append(d); edges_t.append(dur[r, col])
        if (s0 // args.chunk) % 10 == 0:
            print(f"[s2]   edges: {s0+s1-s0}/{n} sources, {len(edges_u):,} directed edges  ({time.time()-t0:.0f}s)")

    u = np.asarray(edges_u, np.int32); v = np.asarray(edges_v, np.int32)
    d = np.asarray(edges_d, np.float32); t = np.asarray(edges_t, np.float32)
    print(f"[s2] directed edges: {len(u):,}  ({time.time()-t0:.0f}s)")

    # Symmetrise for the undirected-PLL prototype: average the two directions where both exist.
    w = {}
    for a, b, dd, tt in zip(u, v, d, t):
        key = (min(a, b), max(a, b))
        w.setdefault(key, []).append((dd, tt))
    eu, ev, ed, et = [], [], [], []
    for (a, b), vals in w.items():
        ed.append(float(np.mean([x[0] for x in vals]))); et.append(float(np.mean([x[1] for x in vals])))
        eu.append(a); ev.append(b)
    eu = np.asarray(eu, np.int32); ev = np.asarray(ev, np.int32)
    ed = np.asarray(ed, np.float32); et = np.asarray(et, np.float32)
    print(f"[s2] undirected edges: {len(eu):,}  (avg degree {2*len(eu)/n:.1f})")

    np.savez(args.out, n=n, lat=lat, lon=lon, u=eu, v=ev, dist=ed, dur=et)
    # C binary: n, m, then (u,v,w) for the two weight kinds side by side
    with open(str(args.out).replace(".npz", ".bin"), "wb") as f:
        f.write(struct.pack("<ii", n, len(eu)))
        for a, b, dd in zip(eu, ev, ed):
            f.write(struct.pack("<iif", int(a), int(b), float(dd)))
    print(f"[s2] wrote {args.out} and .bin  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
