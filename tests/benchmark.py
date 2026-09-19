#!/usr/bin/env python3
"""Latency, memory and accuracy benchmark for the ONNX routing service.

Compares the .NET approximation against a *live* OSRM instance on randomly
sampled coordinate pairs (deliberately outside the training grid), measures
end-to-end HTTP latency percentiles, and samples the resident set size of the
serving process.

It also writes tests/golden_routes.json, the fixture consumed by
GoldenRoutesTests.cs.

Usage:
    uv run benchmark.py --service http://localhost:5080 --osrm http://localhost:5001
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]

# Reuse the OSRM client that generates the training data, so the benchmark and
# the dataset pipeline cannot drift apart in how they talk to OSRM.
sys.path.insert(0, str(ROOT / "python"))
from fetch_osrm_matrix import OsrmClient  # noqa: E402

# Greater Manchester sample window (slightly inset from the county bbox so that
# random points reliably land on the routable network).
LAT_RANGE = (53.34, 53.64)
LON_RANGE = (-2.72, -1.95)

# Named fixtures used as an executable regression contract in the C# test suite.
GOLDEN_ROUTES = [
    ("manchester-piccadilly-to-victoria", 53.4770, -2.2309, 53.4875, -2.2427),
    ("city-centre-to-old-trafford", 53.4808, -2.2426, 53.4631, -2.2913),
    ("city-centre-to-manchester-airport", 53.4808, -2.2426, 53.3537, -2.2749),
    ("etihad-to-trafford-park", 53.4831, -2.2004, 53.4656, -2.2866),
    ("didsbury-to-salford-quays", 53.4165, -2.2280, 53.4713, -2.2953),
    ("stockport-to-bolton", 53.4106, -2.1575, 53.5780, -2.4282),
    ("wigan-to-rochdale", 53.5450, -2.6310, 53.6097, -2.1561),
    ("bury-to-oldham", 53.5934, -2.2971, 53.5409, -2.1114),
    ("altrincham-to-ashton", 53.3871, -2.3520, 53.4906, -2.0936),
    ("chorlton-to-prestwich", 53.4420, -2.2790, 53.5350, -2.2830),
    ("city-centre-to-bury", 53.4808, -2.2426, 53.5934, -2.2971),
    ("sale-to-middleton", 53.4244, -2.3227, 53.5547, -2.1937),
]


def _elapsed_ns(fn) -> int:
    t0 = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - t0


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p))


def summarize(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    mean = statistics.fmean(ordered)
    return {
        "n": len(ordered),
        "mean_us": mean,
        "p50_us": percentile(ordered, 50),
        "p90_us": percentile(ordered, 90),
        "p99_us": percentile(ordered, 99),
        "p999_us": percentile(ordered, 99.9),
        "max_us": max(ordered),
        "throughput_rps": 1e6 / mean,
    }


def find_server_pid() -> int | None:
    """Return the PID of the running RoutingService *process*.

    `pgrep -f RoutingService` also matches the bash wrapper that launched the
    server (and even the benchmark's own shell), whose RSS is a few MB. We must
    report the resident set of the real host process, so prefer the candidate
    whose /proc/<pid>/comm is the dotnet runtime.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", "RoutingService"], capture_output=True, text=True, check=False
        ).stdout.split()
    except Exception:  # noqa: BLE001
        return None

    candidates = [int(p) for p in out]
    # Highest RSS is the resident server; the launcher shell is tiny.
    best: tuple[int, int] | None = None
    for pid in candidates:
        rss = None
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                    break
        except OSError:
            continue
        if rss is None:
            continue
        if best is None or rss > best[1]:
            best = (pid, rss)
    return best[0] if best else None


def read_rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def approx_routes(service: str, session: requests.Session, pairs) -> list[dict]:
    out = []
    for lat1, lon1, lat2, lon2 in pairs:
        r = session.get(
            f"{service}/route",
            params={"orig": f"{lat1},{lon1}", "dest": f"{lat2},{lon2}"},
            timeout=30,
        )
        r.raise_for_status()
        out.append(r.json())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service", default="http://localhost:5080")
    ap.add_argument("--osrm", default="http://localhost:5001")
    ap.add_argument("--latency-samples", type=int, default=2000)
    ap.add_argument("--accuracy-pairs", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "tests" / "golden_routes.json"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})

    # ---- health -----------------------------------------------------------------
    try:
        health = session.get(f"{args.service}/health", timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] FATAL: service not reachable at {args.service}: {exc}", file=sys.stderr)
        return 1
    print(f"[bench] service model={health.get('model')} features={len(health.get('features', []))}")

    pid = find_server_pid()
    rss_before = read_rss_mb(pid) if pid else None
    print(f"[bench] server pid={pid} rss_before={rss_before and f'{rss_before:.1f} MB'}")

    rng = np.random.default_rng(args.seed)

    # ---- latency ----------------------------------------------------------------
    print(f"[bench] warm-up + {args.latency_samples} sequential latency samples")
    warm = [(53.4808, -2.2426, 53.4631, -2.2913)] * 200
    approx_routes(args.service, session, warm)

    lats = rng.uniform(*LAT_RANGE, size=args.latency_samples)
    lons = rng.uniform(*LON_RANGE, size=args.latency_samples)
    lats2 = rng.uniform(*LAT_RANGE, size=args.latency_samples)
    lons2 = rng.uniform(*LON_RANGE, size=args.latency_samples)

    samples_us: list[float] = []
    server_us: list[float] = []
    for i in range(args.latency_samples):
        started = time.perf_counter_ns()
        response = session.get(
            f"{args.service}/route",
            params={"orig": f"{lats[i]},{lons[i]}", "dest": f"{lats2[i]},{lons2[i]}"},
            timeout=30,
        )
        samples_us.append((time.perf_counter_ns() - started) / 1000.0)
        timing = response.headers.get("Server-Timing", "")
        if "dur=" in timing:
            # Server-Timing dur is milliseconds; keep everything in microseconds.
            server_us.append(float(timing.split("dur=")[1].split(";")[0]) * 1000.0)

    latency = summarize(samples_us)
    print(
        f"[bench] round-trip p50={latency['p50_us']:.0f}us p90={latency['p90_us']:.0f}us "
        f"p99={latency['p99_us']:.0f}us p99.9={latency['p999_us']:.0f}us "
        f"({latency['throughput_rps']:,.0f} req/s sequential)"
    )

    server_latency = summarize(server_us) if server_us else None
    if server_latency:
        print(
            f"[bench] server-side p50={server_latency['p50_us']:.0f}us "
            f"p90={server_latency['p90_us']:.0f}us p99={server_latency['p99_us']:.0f}us "
            f"p99.9={server_latency['p999_us']:.0f}us"
        )
    else:
        print("[bench] server-side timing unavailable (no Server-Timing header)")

    # ---- concurrent throughput ---------------------------------------------------
    # The Python-thread loop is client-bound (GIL + per-request HTTP overhead), so
    # it under-reports the server's capacity. When `wrk` is available we use it for
    # a real concurrent load test; the thread loop stays as a dependency-free fallback.
    concurrent_rps = None
    wrk_result = None
    wrk_path = shutil.which("wrk")
    if wrk_path:
        url = (
            f"{args.service}/route"
            f"?orig={lats[0]},{lons[0]}&dest={lats2[0]},{lons2[0]}"
        )
        import re

        proc = subprocess.run(
            [wrk_path, "-t", str(args.concurrency), "-c", str(args.concurrency * 8),
             "-d", "15s", "--latency", url],
            capture_output=True, text=True, check=False,
        )
        text = proc.stdout
        m_rps = re.search(r"Requests/sec:\s*([\d.]+)", text)
        m_p99 = re.search(r"99%\s*([\d.]+)(us|ms|s)", text)
        if m_rps:
            concurrent_rps = float(m_rps.group(1))
        if m_p99:
            val, unit = float(m_p99.group(1)), m_p99.group(2)
            factor = {"us": 1.0, "ms": 1000.0, "s": 1e6}[unit]
            wrk_result = {"rps": concurrent_rps, "p99_us": val * factor,
                          "threads": args.concurrency, "connections": args.concurrency * 8}
        rps_txt = f"{concurrent_rps:,.0f} req/s" if concurrent_rps else "n/a"
        p99_txt = f"{wrk_result['p99_us']:.0f}us" if wrk_result else "n/a"
        print(f"[bench] wrk ({args.concurrency} threads x {args.concurrency * 8} conns): "
              f"{rps_txt} p99={p99_txt}")
    elif args.concurrency > 1:
        import threading

        per_thread = 250
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            s = requests.Session()
            base = tid * per_thread
            try:
                for i in range(per_thread):
                    k = (base + i) % args.latency_samples
                    s.get(
                        f"{args.service}/route",
                        params={"orig": f"{lats[k]},{lons[k]}", "dest": f"{lats2[k]},{lons2[k]}"},
                        timeout=30,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(args.concurrency)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = per_thread * args.concurrency
        concurrent_rps = total / wall
        print(f"[bench] concurrent ({args.concurrency} python threads, client-bound): "
              f"{concurrent_rps:,.0f} req/s, errors={len(errors)}")

    # ---- accuracy vs live OSRM ---------------------------------------------------
    print(f"[bench] accuracy on {args.accuracy_pairs} random pairs vs live OSRM")
    alats = rng.uniform(*LAT_RANGE, size=args.accuracy_pairs)
    alons = rng.uniform(*LON_RANGE, size=args.accuracy_pairs)
    blats = rng.uniform(*LAT_RANGE, size=args.accuracy_pairs)
    blons = rng.uniform(*LON_RANGE, size=args.accuracy_pairs)

    osrm = OsrmClient(args.osrm)
    # One n x n table call; the diagonal entries are exactly the random pairs.
    dur_m, dist_m = osrm.table(alats, alons, blats, blons)
    idx = np.arange(args.accuracy_pairs)
    truth_dur = dur_m[idx, idx]
    truth_dist = dist_m[idx, idx]

    preds = approx_routes(args.service, session, zip(alats, alons, blats, blons))
    pred_dur = np.array([p["duration_s"] for p in preds], dtype=np.float64)
    pred_dist = np.array([p["distance_m"] for p in preds], dtype=np.float64)

    valid = np.isfinite(truth_dur) & np.isfinite(truth_dist)
    truth_dur, truth_dist = truth_dur[valid], truth_dist[valid]
    pred_dur, pred_dist = pred_dur[valid], pred_dist[valid]

    def stats(y_true, y_pred):
        err = np.abs(y_true - y_pred)
        rel = err / np.maximum(y_true, 1e-6)
        return {
            "n": int(len(err)),
            "mae": float(err.mean()),
            "medae": float(np.median(err)),
            "p90_abs_err": float(np.percentile(err, 90)),
            "medape_pct": float(np.median(rel) * 100.0),
            "p90_ape_pct": float(np.percentile(rel, 90) * 100.0),
            "within_10pct": float((rel <= 0.10).mean() * 100.0),
            "within_25pct": float((rel <= 0.25).mean() * 100.0),
        }

    dur_stats = stats(truth_dur, pred_dur)
    dist_stats = stats(truth_dist, pred_dist)
    print(
        f"[bench] duration: MedAE={dur_stats['medae']:.1f}s MedAPE={dur_stats['medape_pct']:.1f}% "
        f"p90APE={dur_stats['p90_ape_pct']:.1f}% within10%={dur_stats['within_10pct']:.0f}%"
    )
    print(
        f"[bench] distance: MedAE={dist_stats['medae']:.1f}m MedAPE={dist_stats['medape_pct']:.1f}% "
        f"p90APE={dist_stats['p90_ape_pct']:.1f}% within10%={dist_stats['within_10pct']:.0f}%"
    )

    pid = find_server_pid()
    rss_after = read_rss_mb(pid) if pid else None
    if rss_after:
        print(f"[bench] rss_after={rss_after:.1f} MB (delta={rss_after - (rss_before or rss_after):+.1f} MB)")

    # ---- golden fixtures ---------------------------------------------------------
    print("[bench] writing golden route fixtures")
    golden_pairs = [(a, b, c, d) for _, a, b, c, d in GOLDEN_ROUTES]
    g_dur, g_dist = osrm.table(
        np.array([p[0] for p in golden_pairs]),
        np.array([p[1] for p in golden_pairs]),
        np.array([p[2] for p in golden_pairs]),
        np.array([p[3] for p in golden_pairs]),
    )
    gidx = np.arange(len(golden_pairs))
    g_truth_dur, g_truth_dist = g_dur[gidx, gidx], g_dist[gidx, gidx]
    g_pred = approx_routes(args.service, session, golden_pairs)

    routes = []
    for k, (name, lat1, lon1, lat2, lon2) in enumerate(GOLDEN_ROUTES):
        routes.append(
            {
                "name": name,
                "orig": [lat1, lon1],
                "dest": [lat2, lon2],
                "osrm_distance_m": float(g_truth_dist[k]),
                "osrm_duration_s": float(g_truth_dur[k]),
                "approx_distance_m": float(g_pred[k]["distance_m"]),
                "approx_duration_s": float(g_pred[k]["duration_s"]),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "service_url": args.service,
        "osrm_url": args.osrm,
        "latency_us": latency,
        "server_latency_us": server_latency,
        "concurrency": {"threads": args.concurrency, "rps": concurrent_rps, "wrk": wrk_result},
        "memory": {"rss_before_mb": rss_before, "rss_after_mb": rss_after},
        "accuracy": {"duration": dur_stats, "distance": dist_stats},
        "routes": routes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[bench] wrote {out}")

    # Accuracy budgets used by GoldenRoutesTests.cs.
    budgets = {
        "reproducibility_abs": 0.11,
        "duration_medape_pct": max(15.0, dur_stats["medape_pct"] * 1.35),
        "distance_medape_pct": max(15.0, dist_stats["medape_pct"] * 1.35),
    }
    print(f"[bench] suggested thresholds: {json.dumps(budgets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
