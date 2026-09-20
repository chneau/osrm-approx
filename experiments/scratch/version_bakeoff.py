#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "lightgbm>=4.7.0",
#   "numpy>=2.5.3",
#   "onnx>=1.23.0",
#   "onnxruntime>=1.30.0",
#   "pandas>=3.0.6",
#   "pyarrow>=25.0.1",
#   "requests>=2.34.2",
# ]
# ///
"""Score the old 63-leaf and shipped 511-leaf ONNX models on a genuinely fair split.

Why a fresh split is needed: the two *shipped* artifacts were trained on different
data, so no existing file is held out for both of them.

    * 511-leaf `server/models/model.onnx` — trained on `samples.parquet` + ALL of
      `offnetwork.parquet` (E5), with E1 inverse-frequency weights.
    * 63-leaf `model_small.onnx` (commit 0357738) — trained on `samples.parquet` only,
      no weights; it predates E5 entirely.

Scoring both on `offnetwork.parquet` therefore flatters the 511-leaf model (it has
seen those exact rows) and is not a fair price. Instead this script draws NEW random
raw-coordinate pairs, labels them with live OSRM `/table`, and scores both models on
those — a split neither artifact has seen. That is also the distribution the API
actually serves (arbitrary raw coordinates, not grid nodes).

Caveat kept explicit: off-network points are out-of-distribution for the 63-leaf
model by construction (it only ever saw grid pairs), which is exactly the point — it
measures what each shipped artifact would do in production, not a controlled
capacity ablation. For the controlled version (identical data, only `num_leaves`
varies) see `price_capacity.py`.

Usage:
    uv run experiments/scratch/version_bakeoff.py --n 3000
    uv run experiments/scratch/version_bakeoff.py --osrm http://localhost:5001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from fetch_osrm_matrix import OsrmClient  # noqa: E402
from train_export_onnx import geometric_features  # noqa: E402

# Same window the benchmark and the off-network sampler use.
LAT_RANGE = (53.34, 53.64)
LON_RANGE = (-2.72, -1.95)
BLOCK = 100  # /table is limited by --max-table-size 10000 -> 100x100 cells per call

BUCKETS = [
    ("<1km", 0.0, 1_000.0),
    ("1-3km", 1_000.0, 3_000.0),
    ("3-10km", 3_000.0, 10_000.0),
    ("10-25km", 10_000.0, 25_000.0),
    (">25km", 25_000.0, float("inf")),
]

MODELS = {
    "old-63leaf": ROOT / "experiments" / "scratch" / "model_small.onnx",
    "shipped-511leaf": ROOT / "server" / "models" / "model.onnx",
}


# Separation bands (metres, great-circle) used by --stratify, copied from
# tests/osrm_vs_onnx.py so the two reports are sampled identically.
HOP_BANDS = [(200, 1_000), (1_000, 3_000), (3_000, 10_000), (10_000, 25_000), (25_000, 60_000)]


def sample_pairs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    o = np.column_stack([rng.uniform(*LAT_RANGE, n), rng.uniform(*LON_RANGE, n)])
    d = np.column_stack([rng.uniform(*LAT_RANGE, n), rng.uniform(*LON_RANGE, n)])
    return o, d


def sample_pairs_stratified(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Even coverage across separation bands: pick an origin, then a destination at a
    random bearing and a distance drawn from each band. Uniform bbox sampling almost
    never produces short trips, which is what makes it flattering (see README)."""
    rng = np.random.default_rng(seed)
    per = max(1, n // len(HOP_BANDS))
    o, d = [], []
    for lo, hi in HOP_BANDS:
        got = 0
        while got < per:
            lat1, lon1 = rng.uniform(*LAT_RANGE), rng.uniform(*LON_RANGE)
            heading = rng.uniform(0.0, 2.0 * np.pi)
            dist_m = rng.uniform(lo, hi)
            lat2 = lat1 + (dist_m * np.cos(heading)) / 111_320.0
            lon2 = lon1 + (dist_m * np.sin(heading)) / (111_320.0 * np.cos(np.radians(lat1)))
            if not (LAT_RANGE[0] <= lat2 <= LAT_RANGE[1] and LON_RANGE[0] <= lon2 <= LON_RANGE[1]):
                continue
            o.append((lat1, lon1))
            d.append((lat2, lon2))
            got += 1
    return np.asarray(o), np.asarray(d)


def osrm_truth(client: OsrmClient, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact (duration_s, distance_m) for each pair, via block-diagonal /table calls."""
    dur = np.full(len(o), np.nan)
    dist = np.full(len(o), np.nan)
    for start in range(0, len(o), BLOCK):
        stop = min(start + BLOCK, len(o))
        dmat, dmat_m = client.table(o[start:stop, 0], o[start:stop, 1],
                                    d[start:stop, 0], d[start:stop, 1])
        idx = np.arange(stop - start)
        dur[start:stop] = np.asarray(dmat)[idx, idx]
        dist[start:stop] = np.asarray(dmat_m)[idx, idx]
    return dur, dist


def load_session(path: Path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    def pick(*needles: str) -> int:
        for i, name in enumerate(names):
            if any(nd in name.lower() for nd in needles):
                return i
        return -1

    di, ti = pick("distance"), pick("duration")
    if di < 0 or ti < 0:  # documented export order: distance_m, duration_s
        di, ti = 0, 1
    return sess, sess.get_inputs()[0].name, di, ti


def predict(path: Path, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sess, input_name, di, ti = load_session(path)
    out = sess.run(None, {input_name: X.astype(np.float32)})
    return np.asarray(out[di]).ravel(), np.asarray(out[ti]).ravel()


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(truth - pred)
    rel = np.where(truth > 1e-6, err / truth, np.nan) * 100.0
    return {
        "n": int(len(truth)),
        "medape": float(np.nanmedian(rel)),
        "mape": float(np.nanmean(rel)),
        "medae": float(np.median(err)),
        "mae": float(err.mean()),
        "within10_pct": float(np.nanmean(rel <= 10.0) * 100.0),
        "bias": float(np.mean(pred - truth)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--stratify", action="store_true",
                    help="sample evenly across separation bands instead of uniform bbox sampling")
    ap.add_argument("--osrm", default="http://localhost:5001")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments" / "scratch" / "version_bakeoff.json")
    args = ap.parse_args()

    t0 = time.time()
    o, d = sample_pairs_stratified(args.n, args.seed) if args.stratify else sample_pairs(args.n, args.seed)
    mode = "stratified" if args.stratify else "uniform"
    print(f"[bakeoff] {len(o):,} fresh {mode} pairs; labelling with live OSRM {args.osrm} ...")
    dur, dist = osrm_truth(OsrmClient(args.osrm), o, d)

    routable = np.isfinite(dur) & np.isfinite(dist) & (dur > 0) & (dist > 0)
    o, d, dur, dist = o[routable], d[routable], dur[routable], dist[routable]
    print(f"[bakeoff] routable pairs: {routable.sum():,} / {args.n:,} ({time.time() - t0:.0f}s)")

    X = geometric_features(o[:, 0], o[:, 1], d[:, 0], d[:, 1]).astype(np.float32)

    payload = {"n_pairs": int(len(o)), "n_routable": int(routable.sum()),
               "sampling": mode, "seed": args.seed, "osrm": args.osrm, "models": {}}

    for label, path in MODELS.items():
        if not path.exists():
            print(f"[bakeoff] missing {path} — skipping {label}")
            continue
        pd_, pt_ = predict(path, X)
        rec = {
            "artifact_mb": round(path.stat().st_size / 1e6, 2),
            "distance": {"overall": metrics(dist, pd_), "by_bucket": {}},
            "duration": {"overall": metrics(dur, pt_), "by_bucket": {}},
        }
        # Both targets are bucketed by OSRM *distance* (never by their own units),
        # matching tests/osrm_vs_onnx.py's stratified breakdown.
        for key, truth, pred in (("distance", dist, pd_), ("duration", dur, pt_)):
            for blabel, lo, hi in BUCKETS:
                mask = (dist >= lo) & (dist < hi)
                if mask.sum() >= 30:
                    rec[key]["by_bucket"][blabel] = metrics(truth[mask], pred[mask])
        payload["models"][label] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n=== held-out raw coordinates ({mode}), labelled by live OSRM ===")
    print(f"{'model':16s} {'target':9s} {'MedAPE':>7s} {'MAE':>9s} {'MedAE':>9s} "
          f"{'<=10%':>7s} {'bias':>9s}")
    for label, rec in payload["models"].items():
        for target in ("distance", "duration"):
            m = rec[target]["overall"]
            print(f"{label:16s} {target:9s} {m['medape']:6.1f}% {m['mae']:9.0f} "
                  f"{m['medae']:9.0f} {m['within10_pct']:6.0f}% {m['bias']:+9.0f}")

    for target in ("distance", "duration"):
        print(f"\n--- {target}: MedAPE by OSRM distance ---")
        print(f"{'model':16s} " + " ".join(f"{b:>8s}" for b, _, _ in BUCKETS))
        for label, rec in payload["models"].items():
            cells = " ".join(
                f"{rec[target]['by_bucket'].get(b, {}).get('medape', float('nan')):8.1f}"
                for b, _, _ in BUCKETS
            )
            print(f"{label:16s} {cells}")

    print(f"\n[bakeoff] wrote {args.out}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
