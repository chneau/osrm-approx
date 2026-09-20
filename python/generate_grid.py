#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy>=2.5.3",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
# ]
# ///
"""Multi-resolution grid generator for Greater Manchester.

Produces candidate origin/destination points that are later fed to OSRM's
/table endpoint.  Resolution is intentionally non-uniform: dense (100-250 m)
in the city core where street topology varies fastest, coarse (500 m-1 km)
towards the fringes of the metropolitan county.

Output: data/processed/grid.parquet  (columns: lat, lon, ring)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Greater Manchester (metropolitan county) bounding box.
BBOX = {"min_lat": 53.315, "max_lat": 53.660, "min_lon": -2.760, "max_lon": -1.890}

# Manchester city centre (Albert Square) - used to drive the multi-resolution rings.
CITY_CENTRE = (53.4808, -2.2426)

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_grid(rings) -> pd.DataFrame:
    """rings: list of (max_radius_km, step_deg) ordered by increasing radius."""
    rows = []
    min_lat, max_lat = BBOX["min_lat"], BBOX["max_lat"]
    min_lon, max_lon = BBOX["min_lon"], BBOX["max_lon"]
    clat, clon = CITY_CENTRE

    for ring_id, (max_radius_km, step_deg) in enumerate(rings):
        # Cover the circle bounding box, then clip to the ring annulus.
        dlat = max_radius_km / 111.32
        dlon = max_radius_km / (111.32 * math.cos(math.radians(clat)))
        lats = np.arange(max(min_lat, clat - dlat), min(max_lat, clat + dlat) + step_deg / 2, step_deg)
        lons = np.arange(max(min_lon, clon - dlon), min(max_lon, clon + dlon) + step_deg / 2, step_deg)

        for la in lats:
            for lo in lons:
                rows.append((round(float(la), 6), round(float(lo), 6), ring_id))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ring"]).drop_duplicates(["lat", "lon"]).reset_index(drop=True)
    dist_km = haversine_m(df["lat"].to_numpy(), df["lon"].to_numpy(), clat, clon) / 1000.0

    keep = np.zeros(len(df), dtype=bool)
    lower = 0.0
    for ring_id, (max_radius_km, _step) in enumerate(rings):
        keep |= (dist_km >= lower) & (dist_km < max_radius_km)
        lower = max_radius_km
    df = df[keep].reset_index(drop=True)
    df["dist_centre_km"] = np.round(haversine_m(df["lat"].to_numpy(), df["lon"].to_numpy(), clat, clon) / 1000.0, 3)
    return df


def build_uniform_grid(bbox, spacing_m: float) -> pd.DataFrame:
    """Regular lat/lon lattice at approximately ``spacing_m`` metres.

    The latitude step is exact; the longitude step is scaled by
    ``cos(mid-latitude)`` so cells are roughly square in metres. The grid is
    aligned to multiples of the step inside the bbox, so the same region and
    spacing always produce the same points.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    dlat = spacing_m / 111_320.0
    dlon = spacing_m / (111_320.0 * math.cos(math.radians(mid_lat)))
    lat0 = math.ceil(min_lat / dlat) * dlat
    lon0 = math.ceil(min_lon / dlon) * dlon
    lats = np.arange(lat0, max_lat + dlat * 1e-6, dlat)
    lons = np.arange(lon0, max_lon + dlon * 1e-6, dlon)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat": np.round(la.ravel(), 6), "lon": np.round(lo.ravel(), 6)})
    df = df.drop_duplicates(["lat", "lon"]).reset_index(drop=True)
    df["ring"] = 0
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "grid.parquet"))
    ap.add_argument("--max-points", type=int, default=3000,
                    help="randomly subsample if the grid is larger (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spacing-m", type=float, default=0.0,
                    help="uniform grid at ~this metric spacing in metres (0 keeps the multi-resolution rings)")
    ap.add_argument("--bbox", default=None,
                    help="'min_lat,min_lon,max_lat,max_lon' (defaults to the built-in Greater Manchester region)")
    # (radius_km, step_deg) -- ~250 m core, ~500 m mid, ~1 km fringe.
    ap.add_argument("--rings", default="3:0.0025,10:0.005,40:0.01")
    args = ap.parse_args()

    if args.bbox:
        vals = [float(v) for v in args.bbox.split(",")]
        if len(vals) != 4 or not (vals[0] < vals[2] and vals[1] < vals[3]):
            raise SystemExit("--bbox must be 'min_lat,min_lon,max_lat,max_lon' with min < max")
        BBOX.update({"min_lat": vals[0], "min_lon": vals[1], "max_lat": vals[2], "max_lon": vals[3]})

    if args.spacing_m and args.spacing_m > 0:
        bbox = (BBOX["min_lat"], BBOX["min_lon"], BBOX["max_lat"], BBOX["max_lon"])
        df = build_uniform_grid(bbox, args.spacing_m)
        print(f"[grid] uniform ~{args.spacing_m:g}m grid over {bbox} -> {len(df):,} points")
    else:
        rings = []
        for part in args.rings.split(","):
            radius, step = part.split(":")
            rings.append((float(radius), float(step)))

        df = build_grid(rings)
        print(f"[grid] raw points: {len(df)}")
        for ring_id, (radius, step) in enumerate(rings):
            n = int((df["ring"] == ring_id).sum())
            approx_m = step * 111_320
            print(f"[grid]   ring {ring_id}: radius<{radius:g}km step={approx_m:.0f}m -> {n} points")

    if args.max_points and args.max_points > 0 and len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=args.seed).sort_values(["lat", "lon"]).reset_index(drop=True)
        print(f"[grid] subsampled to {len(df)} points")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[grid] wrote {len(df)} points -> {args.out}")
    print(f"[grid] pairwise samples: {len(df) ** 2:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
