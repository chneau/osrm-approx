#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy>=2.5.3",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "pyrosm>=0.13.1",
#   "scikit-learn>=1.9.1",
#   "scipy>=1.18.1",
# ]
# ///
"""S2b: can a *real but simplified* OSM driving graph approximate OSRM?

Builds the pyrosm driving network (node-based, no turn restrictions), weights each
segment by great-circle length and a maxspeed-derived duration, then compares
shortest paths to OSRM on the same raw-coordinate test splits used elsewhere.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PBF = Path("/home/c/go/src/github.com/chneau/testing/testing-ml-tte/data/raw/greater-manchester.osm.pbf")
CACHE = Path("/tmp/osm_nodes.parquet"), Path("/tmp/osm_edges.parquet")
BUCKETS = [("<1km", 0, 1_000), ("1-3km", 1_000, 3_000), ("3-10km", 3_000, 10_000), (">25km", 25_000, np.inf)]

SPEED_KMH = {"motorway": 113, "motorway_link": 90, "trunk": 96, "trunk_link": 80, "primary": 64,
             "primary_link": 50, "secondary": 56, "secondary_link": 48, "tertiary": 48, "tertiary_link": 40,
             "unclassified": 40, "residential": 32, "living_street": 10, "service": 16, "road": 32}


def load_network(nodes_p, edges_p):
    if nodes_p.exists() and edges_p.exists():
        nodes = pd.read_parquet(nodes_p); edges = pd.read_parquet(edges_p)
    else:
        from pyrosm import OSM
        osm = OSM(str(PBF))
        nodes, edges = osm.get_network(network_type="driving", nodes=True)
        nodes.to_parquet(nodes_p); edges.to_parquet(edges_p)
    return nodes, edges


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_speed(ms, hw):
    v = np.full(len(ms), np.nan)
    s = ms.fillna("").astype(str).str.strip().str.lower()
    num = pd.to_numeric(s.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    mph = s.str.contains("mph")
    v = np.where(mph, num*1.609344, np.where(num.notna(), num, np.nan))
    base = hw.map(SPEED_KMH).to_numpy(np.float64)
    return np.where(np.isfinite(v) & (v > 0), v, base)


def metrics(y, p):
    e = np.abs(y - p); rel = np.where(y > 1e-6, e / y, np.nan)
    return dict(n=len(y), medape=float(np.nanmedian(rel)*100), mae=float(e.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--drop-residential", action="store_true", help="simplify: keep only >= tertiary")
    args = ap.parse_args()
    t0 = time.time()

    nodes, edges = load_network(*CACHE)
    nlat = nodes.set_index("id")["lat"]; nlon = nodes.set_index("id")["lon"]
    if args.drop_residential:
        keep = edges["highway"].isin(["motorway", "motorway_link", "trunk", "trunk_link", "primary",
                                      "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"])
        edges = edges[keep]
        print(f"[osm] simplified to major roads: {len(edges)} edges")
    lat_u = edges["u"].map(nlat).to_numpy(); lon_u = edges["u"].map(nlon).to_numpy()
    lat_v = edges["v"].map(nlat).to_numpy(); lon_v = edges["v"].map(nlon).to_numpy()
    length = haversine(lat_u, lon_u, lat_v, lon_v)
    speed = parse_speed(edges["maxspeed"], edges["highway"]) / 3.6
    dur = length / np.maximum(speed, 0.5)
    oneway = edges["oneway"].fillna("no").astype(str).str.lower()

    # Compact node ids
    all_ids = pd.unique(pd.concat([edges["u"], edges["v"]]))
    idx = pd.Series(np.arange(len(all_ids)), index=all_ids)
    ui = edges["u"].map(idx).to_numpy(); vi = edges["v"].map(idx).to_numpy()
    n = len(all_ids)
    nl = nodes.set_index("id").loc[all_ids]
    coords = np.column_stack([nl["lat"].to_numpy(), nl["lon"].to_numpy()])

    fwd = ~oneway.eq("-1").to_numpy()      # u->v allowed
    bwd = ~oneway.eq("yes").to_numpy()     # v->u allowed (alternating treated as both)
    rows = np.r_[ui[fwd], vi[bwd]]; cols = np.r_[vi[fwd], ui[bwd]]
    wd = np.r_[length[fwd], length[bwd]]; wt = np.r_[dur[fwd], dur[bwd]]
    Gd = csr_matrix((wd, (rows, cols)), shape=(n, n))
    Gt = csr_matrix((wt, (rows, cols)), shape=(n, n))
    # Keep only the largest weakly-connected component so every pair is routable.
    from scipy.sparse.csgraph import connected_components
    nc, lab = connected_components(Gd + Gd.T, directed=True)
    big = np.argmax(np.bincount(lab))
    keepn = np.where(lab == big)[0]
    remap = -np.ones(n, np.int64); remap[keepn] = np.arange(len(keepn))
    coords = coords[keepn]
    Gd = Gd[keepn][:, keepn]; Gt = Gt[keepn][:, keepn]
    n = len(keepn)
    print(f"[osm] graph n={n:,} edges={len(wd):,} components={nc} (kept largest)  (built {time.time()-t0:.1f}s)")

    scale = np.array([1.0, np.cos(np.radians(53.5))])
    nn = NearestNeighbors(n_neighbors=1).fit(coords*scale)

    for split in ["offnetwork", "offnetwork_short"]:
        off = pd.read_parquet(ROOT / "data" / "processed" / f"{split}.parquet")
        if len(off) > args.sample:
            off = off.sample(n=args.sample, random_state=0).reset_index(drop=True)
        oi = nn.kneighbors(off[["orig_lat", "orig_lon"]].to_numpy()*scale, return_distance=False).ravel()
        di = nn.kneighbors(off[["dest_lat", "dest_lon"]].to_numpy()*scale, return_distance=False).ravel()
        uniq_o = np.unique(oi)
        pos = {v: k for k, v in enumerate(uniq_o)}
        pdrow = np.array([pos[x] for x in oi])
        Dd = dijkstra(Gd, directed=True, indices=uniq_o)   # (n_src, n)
        Dt = dijkstra(Gt, directed=True, indices=uniq_o)
        pred_d = Dd[pdrow, di]; pred_t = Dt[pdrow, di]
        finite = np.isfinite(pred_d) & np.isfinite(pred_t)
        print(f"  coverage (finite routes): {finite.mean()*100:.2f}%")
        off = off[finite].reset_index(drop=True); pred_d = pred_d[finite]; pred_t = pred_t[finite]
        truth_d = off["osrm_distance_m"].to_numpy(); truth_t = off["osrm_duration_s"].to_numpy()
        print(f"\n==== {split} (n={len(off)}, {len(uniq_o)} unique origins) ====")
        for tag, y, p in (("OSM-distance", truth_d, pred_d), ("OSM-duration", truth_t, pred_t)):
            m = metrics(y, p)
            bias = float(np.median(p / np.maximum(y, 1e-6)))
            print(f"  {tag:13s} ALL      MedAPE={m['medape']:6.1f}%  MAE={m['mae']:9.1f}  ratio p50={bias:.3f}")
            for label, lo, hi in BUCKETS:
                mask = (y >= lo) & (y < hi)
                if mask.sum() < 30:
                    continue
                mm = metrics(y[mask], p[mask])
                print(f"  {tag:13s} {label:7s}  MedAPE={mm['medape']:6.1f}%  n={mm['n']}")
    print(f"\n[osm] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
