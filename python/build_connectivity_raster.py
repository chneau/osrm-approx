#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy>=2.5.3",
#   "osmium>=4.3.1",
#   "scipy>=1.18.1",
# ]
# ///
"""Experiment 1 support: static *connectivity / detour* rasters from the OSM extract.

Experiment 0 showed snapping is not the dominant residual: with the exact snap,
off-network short trips stay ~5x worse than on-network, because two raw points
that are close in a straight line can be far apart by road (a river, canal,
railway or motorway forces a detour). Those barriers are static geometry, so this
pre-computes, once and offline, an O(1) lookup for them:

  road_class   int8[ny, nx]   0 none, 1 local, 2 tertiary, 3 secondary,
                              4 primary, 5 motorway/trunk  (highest class in cell)
  water        bool[ny, nx]   waterway in a cell
  rail         bool[ny, nx]   railway in a cell
  motor        bool[ny, nx]   motorway/trunk in a cell
  bar_dist     f32[ny, nx]    metres to the nearest barrier cell (any type)
  meta         JSON string    bbox + cell size (same grid as snap_field.npz)

Query features then sample these along the straight segment between the two raw
points, and look up the class at each endpoint -- all O(1), no graph traversal.

Usage:
    uv run build_connectivity_raster.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import osmium
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BBOX = {"min_lat": 53.315, "max_lat": 53.660, "min_lon": -2.760, "max_lon": -1.890}
LAT0 = 0.5 * (BBOX["min_lat"] + BBOX["max_lat"])
CELL_LAT_DEG = 0.001
EARTH_RADIUS_M = 6_371_008.8
M_PER_DEG_LAT = math.radians(1.0) * EARTH_RADIUS_M

# ordinal road classes (drivable highways only)
CLASS_ORD = {
    "service": 1, "living_street": 1, "residential": 1, "unclassified": 1, "road": 1,
    "tertiary": 2, "tertiary_link": 2,
    "secondary": 3, "secondary_link": 3,
    "primary": 4, "primary_link": 4,
    "motorway": 5, "motorway_link": 5, "trunk": 5, "trunk_link": 5,
}
WATERWAY = {"river", "canal", "stream", "riverbank", "dock", "drain", "ditch"}
RAILWAY = {"rail", "light_rail", "tram", "subway", "narrow_gauge", "monorail"}


class Handler(osmium.SimpleHandler):
    def __init__(self, road_class, water, rail, motor, meta):
        super().__init__()
        self.road_class = road_class
        self.water = water
        self.rail = rail
        self.motor = motor
        self.meta = meta

    def way(self, w):
        tags = w.tags
        cls = CLASS_ORD.get(tags.get("highway", ""))
        is_water = tags.get("waterway") in WATERWAY
        is_rail = tags.get("railway") in RAILWAY
        if cls is None and not is_water and not is_rail:
            return
        prev = None
        for node in w.nodes:
            try:
                loc = node.location
            except osmium.InvalidLocationError:
                prev = None
                continue
            cur = (loc.lat, loc.lon)
            if prev is not None:
                self._mark(prev, cur, cls, is_water, is_rail)
            prev = cur

    def _mark(self, a, b, cls, is_water, is_rail):
        g = self.meta
        mlat = math.radians(0.5 * (a[0] + b[0]))
        dy = math.radians(b[0] - a[0]) * EARTH_RADIUS_M
        dx = math.radians(b[1] - a[1]) * EARTH_RADIUS_M * math.cos(mlat)
        steps = max(1, int(math.hypot(dx, dy) / (0.4 * CELL_LAT_DEG * M_PER_DEG_LAT)))
        for s in range(steps + 1):
            t = s / steps
            lat = a[0] + t * (b[0] - a[0])
            lon = a[1] + t * (b[1] - a[1])
            ic = int((lon - g["min_lon"]) / g["cell_lon"])
            ir = int((lat - g["min_lat"]) / g["cell_lat"])
            if not (0 <= ir < g["ny"] and 0 <= ic < g["nx"]):
                continue
            if cls is not None and cls > self.road_class[ir, ic]:
                self.road_class[ir, ic] = cls
            if is_water:
                self.water[ir, ic] = True
            if is_rail:
                self.rail[ir, ic] = True
            if cls == 5:
                self.motor[ir, ic] = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", default="/home/c/go/src/github.com/chneau/testing/testing-ml-tte/data/raw/greater-manchester.osm.pbf")
    ap.add_argument("--cell-lat", type=float, default=CELL_LAT_DEG)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "connectivity.npz"))
    args = ap.parse_args()

    cell_lat = args.cell_lat
    cell_lon = cell_lat / math.cos(math.radians(LAT0))
    nx = int(math.ceil((BBOX["max_lon"] - BBOX["min_lon"]) / cell_lon))
    ny = int(math.ceil((BBOX["max_lat"] - BBOX["min_lat"]) / cell_lat))
    cell_m = cell_lat * M_PER_DEG_LAT
    meta = {"min_lat": BBOX["min_lat"], "min_lon": BBOX["min_lon"],
            "cell_lat": cell_lat, "cell_lon": cell_lon, "ny": ny, "nx": nx, "cell_m": cell_m}
    print(f"[conn] raster {ny} x {nx} = {ny * nx:,} cells of {cell_m:.1f} m")

    road_class = np.zeros((ny, nx), dtype=np.int8)
    water = np.zeros((ny, nx), dtype=bool)
    rail = np.zeros((ny, nx), dtype=bool)
    motor = np.zeros((ny, nx), dtype=bool)
    print(f"[conn] parsing {args.pbf} ...")
    Handler(road_class, water, rail, motor, meta).apply_file(args.pbf, locations=True)
    print(f"[conn] road cells {int((road_class > 0).sum()):,}  water {int(water.sum()):,}  "
          f"rail {int(rail.sum()):,}  motorway {int(motor.sum()):,}")

    barrier = water | rail | motor
    bar_dist_cells = ndimage.distance_transform_edt(~barrier)
    bar_dist = (bar_dist_cells * cell_m).astype(np.float32)
    print(f"[conn] barrier cells {int(barrier.sum()):,}; bar_dist p50={np.median(bar_dist):.0f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, road_class=road_class, water=water, rail=rail, motor=motor,
                        bar_dist=bar_dist, meta=json.dumps(meta))
    print(f"[conn] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
