#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Measure steady-state memory of the compiled-tree server for a given model.bin.

Reproduces the E11 methodology (see IMPROVEMENTS.md): launch the built server with
`MODEL_PATH` pointing at a `model.bin`, warm it with synthetic requests, let it
settle, then sample `/proc/<pid>/smaps_rollup` and `/proc/<pid>/status`.

The server defaults to the model next to the executable, so `MODEL_PATH` is the
only thing that changes between runs — same binary, same GC settings, same
feature code. That isolates the cost of the tree table itself.

Usage:
    uv run experiments/scratch/measure_model_rss.py --label small \
        --model experiments/scratch/model_small.bin
    uv run experiments/scratch/measure_model_rss.py --label shipped \
        --model server/models/model.bin --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_BIN = ROOT / "server" / "bin" / "Release" / "net10.0" / "RoutingService"

# Same Greater Manchester window the benchmark and training pipeline use.
LAT_RANGE = (53.34, 53.64)
LON_RANGE = (-2.72, -1.95)


def smaps_rollup(pid: int) -> dict[str, float]:
    """Read the kernel's rolled-up smaps: Pss, Private_Dirty, Rss (all kB)."""
    fields: dict[str, float] = {}
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return fields
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.strip().split()
        if value and value[0].replace(".", "", 1).isdigit():
            fields[key.strip()] = float(value[0]) / 1024.0  # kB -> MB
    return fields


def status_fields(pid: int) -> dict[str, float]:
    fields: dict[str, float] = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return fields
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() in {"VmRSS", "VmHWM", "VmPeak"}:
            fields[key.strip()] = float(rest.strip().split()[0]) / 1024.0
    return fields


def get(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def wait_healthy(base: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get(f"{base}/health") is not None:
            return True
        time.sleep(0.25)
    return False


def warm(base: str, n: int, seed: int, concurrency: int) -> int:
    """Hit /route concurrently, so the GC sees the allocation pattern real load
    produces (a sequential loop under-loads it and under-reports the heap)."""
    rng = random.Random(seed)
    pairs = [
        (rng.uniform(*LAT_RANGE), rng.uniform(*LON_RANGE),
         rng.uniform(*LAT_RANGE), rng.uniform(*LON_RANGE))
        for _ in range(n)
    ]

    def one(pair: tuple[float, float, float, float]) -> bool:
        lat1, lon1, lat2, lon2 = pair
        return get(f"{base}/route?orig={lat1:.6f},{lon1:.6f}&dest={lat2:.6f},{lon2:.6f}") is not None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return sum(pool.map(one, pairs))


def run_once(label: str, model: Path, port: int, warm_n: int, seed: int, concurrency: int) -> dict:
    base = f"http://localhost:{port}"
    env = {**os.environ, "MODEL_PATH": str(model), "ASPNETCORE_URLS": base}
    proc = subprocess.Popen(
        [str(SERVER_BIN)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_healthy(base):
            raise SystemExit(f"server for '{label}' never became healthy")
        pid = proc.pid
        health = get(f"{base}/health") or {}

        # Cold reading: taken as soon as the model is loaded and before a single
        # request, so it reflects the load path itself. The warm reading below includes
        # whatever the GC chose to keep committed during warm-up, which can mask it.
        time.sleep(0.5)
        cold = {**smaps_rollup(pid), **status_fields(pid)}

        served = warm(base, warm_n, seed, concurrency)
        time.sleep(2.0)  # let the GC settle so we read steady state, not warm-up

        samples = []
        for _ in range(3):
            samples.append({**smaps_rollup(pid), **status_fields(pid)})
            time.sleep(1.0)

        def med(key: str) -> float | None:
            vals = [s[key] for s in samples if key in s]
            return round(statistics.median(vals), 1) if vals else None

        return {
            "label": label,
            "model": str(model.relative_to(ROOT)),
            "artifact_mb": round(model.stat().st_size / 1e6, 2),
            "loaded": health.get("model"),
            "warm_requests_ok": served,
            "cold_Rss_mb": round(cold.get("Rss", 0.0), 1),
            "cold_Pss_mb": round(cold.get("Pss", 0.0), 1),
            "Pss_mb": med("Pss"),
            "Rss_mb": med("Rss"),
            "Private_Dirty_mb": med("Private_Dirty"),
            "Anonymous_mb": med("Anonymous"),
            "VmRSS_mb": med("VmRSS"),
            "VmHWM_mb": med("VmHWM"),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--port", type=int, default=5081)
    ap.add_argument("--warm", type=int, default=3000,
                    help="total requests sent during warm-up (default 3000)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = args.model if args.model.is_absolute() else (ROOT / args.model)
    if not SERVER_BIN.exists():
        raise SystemExit(f"{SERVER_BIN} not found — run `npm run build` first.")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    label = args.label or model.stem
    results = [
        run_once(label, model, args.port + i, args.warm, args.seed + i, args.concurrency)
        for i in range(args.repeats)
    ]
    for r in results:
        print(json.dumps(r, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
