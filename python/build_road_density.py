#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "osmium>=3.7",
# ]
# ///
"""E4 (see IMPROVEMENTS.md): static road-density raster from the OSM extract.

Coordinates alone cannot tell an urban corridor from open countryside. This
pre-computes, once and offline, the total drivable-road length per ~250 m cell,
so a query can look up its local road density in O(1) -- no per-query graph
traversal, consistent with the service's design.

Output: data/processed/road_density.npz
  density  float32[ny, nx]   road metres per cell
  meta     JSON string       bbox + cell size (degrees)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import osmium

ROOT = Path(__file__).resolve().parents[1]
BBOX = {"min_lat": 53.315, "max_lat": 53.660, "min_lon": -2.760, "max_lon": -1.890}
CELL_DEG = 0.0025  # ~278 m north-south at this latitude

# Drivable highway classes (roughly what /opt/car.lua accepts).
DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link", "road", "service",
}
EARTH_RADIUS_M = 6_371_008.8


class RoadDensity(osmium.SimpleHandler):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.density = np.zeros((grid["ny"], grid["nx"]), dtype=np.float64)
        self.ways = 0
        self.segments = 0

    def way(self, w):
        tags = w.tags
        if tags.get("highway") not in DRIVABLE:
            return
        self.ways += 1
        prev = None
        for node in w.nodes:
            try:
                loc = node.location
            except osmium.InvalidLocationError:  # node missing from extract
                prev = None
                continue
            lat, lon = loc.lat, loc.lon
            if prev is not None:
                self._add_segment(prev[0], prev[1], lat, lon)
                self.segments += 1
            prev = (lat, lon)

    def _add_segment(self, lat1, lon1, lat2, lon2):
        g = self.grid
        # Segment length (equirectangular is fine at ~300 m scale).
        mlat = math.radians((lat1 + lat2) / 2.0)
        dy = math.radians(lat2 - lat1) * EARTH_RADIUS_M
        dx = math.radians(lon2 - lon1) * EARTH_RADIUS_M * math.cos(mlat)
        length = math.hypot(dx, dy)
        mx, my = (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0
        ix = int((my - g["min_lon"]) / g["cell"])
        iy = int((mx - g["min_lat"]) / g["cell"])
        if 0 <= ix < g["nx"] and 0 <= iy < g["ny"]:
            self.density[iy, ix] += length


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", default="/home/c/go/src/github.com/chneau/testing/testing-ml-tte/data/raw/greater-manchester.osm.pbf")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "road_density.npz"))
    args = ap.parse_args()

    nx = int(math.ceil((BBOX["max_lon"] - BBOX["min_lon"]) / CELL_DEG))
    ny = int(math.ceil((BBOX["max_lat"] - BBOX["min_lat"]) / CELL_DEG))
    grid = {"nx": nx, "ny": ny, "cell": CELL_DEG, **BBOX}
    print(f"[density] raster {ny} x {nx} cells of {CELL_DEG}deg")

    handler = RoadDensity(grid)
    print(f"[density] parsing {args.pbf} (building node location index, this takes a minute)...")
    handler.apply_file(args.pbf, locations=True)
    print(f"[density] {handler.ways:,} drivable ways, {handler.segments:,} segments")

    density = handler.density.astype(np.float32)
    meta = {"min_lat": BBOX["min_lat"], "min_lon": BBOX["min_lon"], "cell": CELL_DEG,
            "ny": ny, "nx": nx}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, density=density, meta=json.dumps(meta))
    occupied = float((density > 0).mean())
    print(f"[density] cells with roads: {occupied * 100:.1f}%  "
          f"p50={np.median(density[density > 0]):.0f}m p99={np.percentile(density[density > 0], 99):.0f}m")
    print(f"[density] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
