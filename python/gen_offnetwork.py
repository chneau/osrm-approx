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
"""E5 (see IMPROVEMENTS.md): ground-truth pairs for *off-network* coordinates.

The shipped grid only contains points that sit on the road network. The service,
however, is handed arbitrary raw coordinates (which OSRM would snap). This script
samples random coordinates inside the Greater Manchester bounding box, asks OSRM
to snap+route them via /table, and stores the labels.

Output: data/processed/offnetwork.parquet (same schema as samples.parquet).

Usage:
    uv run gen_offnetwork.py --url http://localhost:5001 --points 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_osrm_matrix import OsrmClient

ROOT = Path(__file__).resolve().parents[1]
BBOX = {"min_lat": 53.315, "max_lat": 53.660, "min_lon": -2.760, "max_lon": -1.890}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:5001")
    ap.add_argument("--points", type=int, default=400, help="random off-network coordinates")
    ap.add_argument("--block", type=int, default=25, help="origins x destinations per /table block")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "offnetwork.parquet"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lat = rng.uniform(BBOX["min_lat"], BBOX["max_lat"], size=args.points)
    lon = rng.uniform(BBOX["min_lon"], BBOX["max_lon"], size=args.points)
    print(f"[offnet] {args.points} random coordinates in bbox "
          f"({BBOX['min_lat']:.4f},{BBOX['min_lon']:.4f})-({BBOX['max_lat']:.4f},{BBOX['max_lon']:.4f})")

    client = OsrmClient(args.url)

    o_lat, o_lon, d_lat, d_lon = [], [], [], []
    dur_all, dist_all = [], []
    blocks = 0
    for s0 in range(0, args.points, args.block):
        s1 = min(s0 + args.block, args.points)
        for t0 in range(0, args.points, args.block):
            t1 = min(t0 + args.block, args.points)
            dur, dist = client.table(lat[s0:s1], lon[s0:s1], lat[t0:t1], lon[t0:t1])
            ok = np.isfinite(dur) & np.isfinite(dist) & (dist >= 1.0) & (dur > 0.0)
            # Drop pairs that snapped onto the same edge position (0 m).
            g_src = s0 + np.arange(s1 - s0)
            g_dst = t0 + np.arange(t1 - t0)
            ok &= g_src[:, None] != g_dst[None, :]
            if not ok.any():
                continue
            rr, cc = np.nonzero(ok)
            o_lat.append(lat[s0:s1][rr])
            o_lon.append(lon[s0:s1][rr])
            d_lat.append(lat[t0:t1][cc])
            d_lon.append(lon[t0:t1][cc])
            dur_all.append(dur[ok])
            dist_all.append(dist[ok])
            blocks += 1

    if not dur_all:
        print("[offnet] FATAL: no routable pairs returned")
        return 1

    samples = pd.DataFrame(
        {
            "orig_lat": np.concatenate(o_lat).astype(np.float32),
            "orig_lon": np.concatenate(o_lon).astype(np.float32),
            "dest_lat": np.concatenate(d_lat).astype(np.float32),
            "dest_lon": np.concatenate(d_lon).astype(np.float32),
            "osrm_distance_m": np.concatenate(dist_all).astype(np.float32),
            "osrm_duration_s": np.concatenate(dur_all).astype(np.float32),
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(args.out, index=False)
    d = samples["osrm_distance_m"]
    print(f"[offnet] wrote {len(samples):,} pairs from {blocks} blocks -> {args.out}")
    print(f"[offnet] distance_m min={d.min():.0f} p50={d.median():.0f} max={d.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
