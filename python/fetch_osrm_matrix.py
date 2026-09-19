#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=17.0",
#   "requests>=2.32",
# ]
# ///
"""Chunked OSRM /table caller -> ground-truth static (duration_s, distance_m).

Reads the multi-resolution grid produced by ``generate_grid.py``, queries a
running OSRM backend in rectangular source/destination chunks, and stores every
routable ordered pair in a Parquet sample set.

Output: data/processed/samples.parquet
  columns: orig_lat orig_lon dest_lat dest_lon osrm_distance_m osrm_duration_s

Usage:
    docker compose --profile serve up -d osrm-routed
    uv run fetch_osrm_matrix.py --url http://localhost:5001
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def _coord_string(lat: np.ndarray, lon: np.ndarray) -> str:
    """OSRM URL coordinates are '{lon},{lat}' -- longitude first."""
    return ";".join(f"{b:.6f},{a:.6f}" for a, b in zip(lat, lon))


class OsrmClient:
    """Thin OSRM /table client that survives chunking and transient failures.

    Only GET is used: this OSRM build ignores form-encoded body parameters
    (``annotations`` in a POST body is silently dropped), while the URL query
    string is honoured and comfortably accepts ~25 kB of coordinates.
    """

    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3, max_split_depth: int = 8):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.max_split_depth = max_split_depth
        self._local = threading.local()

    def _session(self) -> requests.Session:
        # Thread-local sessions: `nearest` fans out across a thread pool and
        # requests.Session is not safe to share for concurrent writes.
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _dispatch(self, n_src: int, n_dst: int, coords: str) -> dict:
        # The query string is assembled verbatim: OSRM splits on ';' and ','
        # and does not decode percent-escapes before doing so, so letting
        # requests encode them (%3B / %2C) silently yields all-zero matrices.
        sources = ";".join(str(i) for i in range(n_src))
        destinations = ";".join(str(i) for i in range(n_src, n_src + n_dst))
        url = (
            f"{self.base_url}/table/v1/driving/{coords}"
            f"?sources={sources}&destinations={destinations}&annotations=duration,distance"
        )
        r = self._session().get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM error: {data.get('code')} {data.get('message')}")
        return data

    def nearest(self, lats, lons, workers: int = 16) -> list[dict]:
        """Snap every point to the road network.

        OSRM's /nearest endpoint accepts exactly one coordinate per request
        ("Only one input coordinate is supported"), so this fans out over a
        thread pool rather than batching coordinates.
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)

        def one(i: int) -> dict:
            url = f"{self.base_url}/nearest/v1/driving/{lons[i]:.6f},{lats[i]:.6f}?number=1"
            last: Exception | None = None
            for attempt in range(self.retries):
                try:
                    r = self._session().get(url, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    if data.get("code") != "Ok":
                        raise RuntimeError(f"{data.get('code')}: {data.get('message')}")
                    return data["waypoints"][0]
                except Exception as exc:  # noqa: BLE001 - retried below
                    last = exc
                    time.sleep(0.3 * (attempt + 1))
            raise RuntimeError(f"/nearest failed for point {i} ({lats[i]},{lons[i]}): {last}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, range(len(lats))))

    def table(self, sources_lat, sources_lon, dests_lat, dests_lon, _depth: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return (duration_s, distance_m) matrices of shape (len(sources), len(dests)).

        If a request fails outright (e.g. URI too long) the destination block is
        bisected and retried, so the caller never has to pick a safe chunk size.
        """
        sources_lat = np.asarray(sources_lat)
        sources_lon = np.asarray(sources_lon)
        dests_lat = np.asarray(dests_lat)
        dests_lon = np.asarray(dests_lon)
        n_src, n_dst = len(sources_lat), len(dests_lat)

        coords = _coord_string(
            np.concatenate([sources_lat, dests_lat]),
            np.concatenate([sources_lon, dests_lon]),
        )

        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                data = self._dispatch(n_src, n_dst, coords)
                dur = np.asarray(data["durations"], dtype=np.float64)
                dist = np.asarray(data["distances"], dtype=np.float64)
                if dur.shape != (n_src, n_dst) or dist.shape != (n_src, n_dst):
                    raise RuntimeError(f"unexpected matrix shape {dur.shape} vs {(n_src, n_dst)}")
                return dur, dist
            except Exception as exc:  # noqa: BLE001 - retried/split below
                last_err = exc
                time.sleep(0.5 * (attempt + 1))

        if _depth >= self.max_split_depth or n_dst <= 1:
            raise RuntimeError(f"OSRM /table failed after {self.retries} attempts: {last_err}")

        mid = n_dst // 2
        d_left, m_left = self.table(sources_lat, sources_lon, dests_lat[:mid], dests_lon[:mid], _depth + 1)
        d_right, m_right = self.table(sources_lat, sources_lon, dests_lat[mid:], dests_lon[mid:], _depth + 1)
        return (
            np.concatenate([d_left, d_right], axis=1),
            np.concatenate([m_left, m_right], axis=1),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:5001", help="OSRM base URL")
    ap.add_argument("--grid", default=str(ROOT / "data" / "processed" / "grid.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "samples.parquet"))
    ap.add_argument("--max-coords", type=int, default=1200, help="sources + destinations per HTTP request")
    ap.add_argument("--max-snap-m", type=float, default=250.0, help="drop grid points further than this from a road")
    ap.add_argument("--include-self", action="store_true", help="keep A->A pairs (all zeros)")
    args = ap.parse_args()

    grid = pd.read_parquet(args.grid)
    lat = grid["lat"].to_numpy(dtype=np.float64)
    lon = grid["lon"].to_numpy(dtype=np.float64)
    n = len(grid)
    print(f"[matrix] grid points: {n} -> {n * n:,} ordered pairs")

    client = OsrmClient(args.url)

    # Points far from any road snap to a distant boundary edge; keeping them
    # would inject degenerate 0 m / 0 s pairs into the training set.
    try:
        waypoints = client.nearest(lat, lon)
    except Exception as exc:  # noqa: BLE001
        print(f"[matrix] FATAL: cannot reach OSRM at {args.url}: {exc}", file=sys.stderr)
        return 1

    snap_m = np.array([float(w["distance"]) for w in waypoints])
    keep = snap_m <= args.max_snap_m
    if not keep.all():
        print(
            f"[matrix] dropped {int((~keep).sum())} off-network points "
            f"(snap > {args.max_snap_m:.0f} m; worst {snap_m.max():.0f} m)"
        )
    lat, lon = lat[keep], lon[keep]
    n = len(lat)
    print(f"[matrix] routable grid points: {n} -> {n * n:,} ordered pairs")

    # Sanity probe on two points guaranteed to lie on the network: the grid
    # points closest to the city centre and to Manchester Airport.
    def _nearest_idx(target_lat: float, target_lon: float) -> int:
        return int(np.argmin((lat - target_lat) ** 2 + (lon - target_lon) ** 2))

    a = _nearest_idx(53.4808, -2.2426)
    b = _nearest_idx(53.3537, -2.2749)
    try:
        probe_dur, probe_dist = client.table(lat[a : a + 1], lon[a : a + 1], lat[b : b + 1], lon[b : b + 1])
        probe_s, probe_m = float(probe_dur.ravel()[0]), float(probe_dist.ravel()[0])
        print(f"[matrix] probe (city centre -> airport) = {probe_s:.1f}s / {probe_m:.1f}m")
        if not np.isfinite(probe_s) or probe_m < 10_000.0:
            raise RuntimeError(
                f"probe returned a degenerate route ({probe_s}s / {probe_m}m) -- "
                "check coordinate order (OSRM expects lon,lat)"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[matrix] FATAL: OSRM sanity probe failed: {exc}", file=sys.stderr)
        return 1

    # Split the request budget between sources and destinations, favouring many
    # sources (OSRM performs one forward search per source, so wide source
    # blocks amortise the per-destination cost).
    src_chunk = max(1, min(args.max_coords // 2, args.max_coords - 1))
    dst_chunk = args.max_coords - src_chunk
    n_requests = ((n + src_chunk - 1) // src_chunk) * ((n + dst_chunk - 1) // dst_chunk)
    print(f"[matrix] chunking: {src_chunk} sources x {dst_chunk} destinations -> ~{n_requests} requests")

    o_lat, o_lon, d_lat, d_lon = [], [], [], []
    dur_all, dist_all = [], []
    done = 0
    started = time.time()

    for s0 in range(0, n, src_chunk):
        s1 = min(s0 + src_chunk, n)
        sl, so = lat[s0:s1], lon[s0:s1]
        for t0 in range(0, n, dst_chunk):
            t1 = min(t0 + dst_chunk, n)
            dur, dist = client.table(sl, so, lat[t0:t1], lon[t0:t1])

            # Sub-metre pairs are points that snapped onto the same edge position.
            ok = np.isfinite(dur) & np.isfinite(dist) & (dist >= 1.0) & (dur > 0.0)
            if not args.include_self and s0 < t1 and t0 < s1:
                # Drop A->A pairs, comparing *global* grid indices.
                g_src = s0 + np.arange(s1 - s0)
                g_dst = t0 + np.arange(t1 - t0)
                ok &= g_src[:, None] != g_dst[None, :]
            if not ok.any():
                continue

            rr, cc = np.nonzero(ok)
            # Row/column indices address the source block and destination block
            # directly. Do NOT index a np.repeat()-ed source array here: that
            # array is laid out per-source, so a row index would collapse every
            # origin onto the first source of the chunk.
            o_lat.append(sl[rr])
            o_lon.append(so[rr])
            d_lat.append(lat[t0:t1][cc])
            d_lon.append(lon[t0:t1][cc])
            dur_all.append(dur[ok])
            dist_all.append(dist[ok])

            done += 1
            if done % 8 == 0 or done == n_requests:
                elapsed = time.time() - started
                pct = 100.0 * done / max(n_requests, 1)
                print(f"[matrix] {done}/{n_requests} requests ({pct:5.1f}%) elapsed={elapsed:6.1f}s")

    if not dur_all:
        print("[matrix] FATAL: no routable pairs returned", file=sys.stderr)
        return 1

    o_lat_c = np.concatenate(o_lat)
    o_lon_c = np.concatenate(o_lon)
    d_lat_c = np.concatenate(d_lat)
    d_lon_c = np.concatenate(d_lon)
    dur_c = np.concatenate(dur_all)
    dist_c = np.concatenate(dist_all)

    # Every grid point appears as an origin and as a destination in a full
    # all-pairs sweep, so both coordinate sets must cover the whole grid.
    # This catches row/column index misalignment between the matrix and the
    # coordinates -- a failure mode that otherwise produces plausible-looking
    # but meaningless labels.
    def _n_unique_points(la: np.ndarray, lo: np.ndarray) -> int:
        pairs = np.stack([np.round(la.astype(np.float64), 6), np.round(lo.astype(np.float64), 6)], axis=1)
        return len(np.unique(pairs, axis=0))

    n_orig = _n_unique_points(o_lat_c, o_lon_c)
    n_dest = _n_unique_points(d_lat_c, d_lon_c)
    if n_orig != n or n_dest != n:
        print(
            f"[matrix] FATAL: coordinate misalignment -- {n_orig} distinct origins and "
            f"{n_dest} distinct destinations for {n} grid points",
            file=sys.stderr,
        )
        return 1

    samples = pd.DataFrame(
        {
            "orig_lat": o_lat_c.astype(np.float32),
            "orig_lon": o_lon_c.astype(np.float32),
            "dest_lat": d_lat_c.astype(np.float32),
            "dest_lon": d_lon_c.astype(np.float32),
            "osrm_distance_m": dist_c.astype(np.float32),
            "osrm_duration_s": dur_c.astype(np.float32),
        }
    )

    # Physical invariant: two points a few hundred metres apart cannot be
    # 20 km apart by road. Guards against silently mislabelled pairs.
    ce = 6_371_008.8
    rlat1, rlon1 = np.radians(o_lat_c), np.radians(o_lon_c)
    rlat2, rlon2 = np.radians(d_lat_c), np.radians(d_lon_c)
    a_h = (
        np.sin((rlat2 - rlat1) / 2) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin((rlon2 - rlon1) / 2) ** 2
    )
    hav = 2 * ce * np.arcsin(np.sqrt(np.clip(a_h, 0, 1)))
    close = hav < 250.0
    if close.any():
        med = float(np.median(dist_c[close]))
        if med > 2_000.0:
            print(
                f"[matrix] FATAL: {int(close.sum())} pairs closer than 250 m have a median "
                f"road distance of {med:.0f} m -- labels look misaligned",
                file=sys.stderr,
            )
            return 1
        print(f"[matrix] sanity: {int(close.sum())} pairs <250 m apart, median road distance {med:.0f} m")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(args.out, index=False)

    d = samples["osrm_distance_m"]
    t = samples["osrm_duration_s"]
    print(f"[matrix] wrote {len(samples):,} routable pairs -> {args.out}")
    print(f"[matrix] distance_m  min={d.min():10.1f} p50={d.median():10.1f} max={d.max():10.1f}")
    print(f"[matrix] duration_s  min={t.min():10.1f} p50={t.median():10.1f} max={t.max():10.1f}")
    print(f"[matrix] total wall time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
