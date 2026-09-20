#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy>=2.5.3",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "requests>=2.34.2",
# ]
# ///
"""Experiment 0 support: the *exact* OSRM snap for every endpoint (offline).

The service is handed raw coordinates; OSRM snaps them to an edge before routing.
This calls `/nearest` once per unique endpoint (build-time, no per-query cost) and
records where OSRM itself put each point, so the experiment can ask: *if the model
knew the exact snap, would the short-trip error disappear?* That is the ceiling a
real snapper could reach.

Output: data/processed/snap_endpoints.parquet
  columns: lat lon snap_lat snap_lon snap_dist_m de_m dn_m
  (de_m/dn_m are the east/north displacement, metres, from raw -> snapped.)

Usage:
    uv run snap_endpoints.py --url http://localhost:5001 --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_osrm_matrix import OsrmClient

ROOT = Path(__file__).resolve().parents[1]
M_PER_DEG_LAT = 110_540.0


def collect_endpoints(frames) -> pd.DataFrame:
    """Union of (lat, lon) across od-pair frames and plain point frames."""
    keys = []
    for df in frames:
        cols = set(df.columns)
        for a, b in (("orig_lat", "orig_lon"), ("dest_lat", "dest_lon")):
            if a in cols:
                keys.append(df[[a, b]].rename(columns={a: "lat", b: "lon"}))
        if "lat" in cols and "lon" in cols and "orig_lat" not in cols:
            keys.append(df[["lat", "lon"]])
    pts = pd.concat(keys, ignore_index=True).drop_duplicates()
    return pts.sort_values(["lat", "lon"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--off", default=str(ROOT / "data" / "processed" / "offnetwork_short.parquet"))
    ap.add_argument("--grid", default=str(ROOT / "data" / "processed" / "grid.parquet"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "snap_endpoints.parquet"))
    args = ap.parse_args()

    frames = [pd.read_parquet(args.off)]
    if args.grid and Path(args.grid).exists():
        g = pd.read_parquet(args.grid)
        if {"lat", "lon"} <= set(g.columns):
            frames.append(g[["lat", "lon"]])
    pts = collect_endpoints(frames)
    lat = pts["lat"].to_numpy(np.float64)
    lon = pts["lon"].to_numpy(np.float64)
    print(f"[snap] {len(pts):,} unique endpoints to snap")

    client = OsrmClient(args.url)
    waypoints = client.nearest(lat, lon, workers=args.workers)

    snap_lat = np.empty_like(lat)
    snap_lon = np.empty_like(lon)
    snap_dist = np.empty_like(lat)
    for i, w in enumerate(waypoints):
        loc = w["location"]  # [lon, lat]
        snap_lon[i] = loc[0]
        snap_lat[i] = loc[1]
        snap_dist[i] = float(w["distance"])

    dn = (snap_lat - lat) * M_PER_DEG_LAT
    de = (snap_lon - lon) * M_PER_DEG_LAT * np.cos(np.radians(lat))
    # /nearest reports its own straight-line snap distance; cross-check ours.
    ours = np.hypot(de, dn)
    err = np.abs(ours - snap_dist)
    print(f"[snap] snap_dist p50={np.median(snap_dist):.1f} m  p90={np.percentile(snap_dist,90):.1f} m")
    print(f"[snap] our displacement vs OSRM distance: max|Δ|={err.max():.2f} m  (sanity, should be ~0)")

    out = pd.DataFrame({
        "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
        "snap_lat": snap_lat.astype(np.float64), "snap_lon": snap_lon.astype(np.float64),
        "snap_dist_m": snap_dist.astype(np.float32),
        "de_m": de.astype(np.float32), "dn_m": dn.astype(np.float32),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[snap] wrote {len(out):,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
