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
#   "scikit-learn>=1.9.1",
# ]
# ///
"""Price the 63 -> 511 leaf capacity bump in accuracy, for the memory it costs.

Companion to the RSS measurement in `measure_model_rss.py`: the 511-leaf model costs
~30 MB more RSS than the 63-leaf one, so what does that 30 MB actually buy?

Protocol (same as `experiments/try_s1_fair.py`, so the numbers are comparable with
the published S1/GBM table): hold out 20% of `offnetwork.parquet` (raw coordinates,
i.e. what the API is handed), train on `samples.parquet` + the other 80%, and score
on the unseen 20%.

Two subtleties that decide whether this measurement is honest:

1. The two *shipped* artifacts cannot be compared directly. The 511-leaf `model.onnx`
   was trained on all of `offnetwork.parquet` (E5) while the 63-leaf one (commit
   0357738) predates E5 and never saw it, so scoring both there flatters the 511-leaf
   model. That contaminated head-to-head is reported at the end, clearly labelled, as
   a demonstration rather than a result.
2. The controlled comparison retrains BOTH capacities with identical data, weights,
   params and seeds, so `num_leaves` is the only variable. That isolates the capacity
   effect from the E1/E5 data effects that also landed between the two commits.

Usage:
    uv run experiments/scratch/price_capacity.py
    uv run experiments/scratch/price_capacity.py --leaves 63 127 255 511
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from train_export_onnx import geometric_features, inverse_frequency_weights  # noqa: E402

import lightgbm as lgb  # noqa: E402

PROC = ROOT / "data" / "processed"
BUCKETS = [
    ("<1km", 0.0, 1_000.0),
    ("1-3km", 1_000.0, 3_000.0),
    ("3-10km", 3_000.0, 10_000.0),
    ("10-25km", 10_000.0, 25_000.0),
    (">25km", 25_000.0, float("inf")),
]

# Identical to the shipped config (E5/E7) — only num_leaves varies between runs.
BASE_PARAMS = dict(
    objective="regression", metric="mae", learning_rate=0.08, n_estimators=400,
    min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
    verbose=-1, num_threads=16,
)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = np.abs(y - p)
    rel = np.where(y > 1e-6, err / y, np.nan)
    return {
        "n": int(len(y)),
        "medape": float(np.nanmedian(rel) * 100.0),
        "mape": float(np.nanmean(rel) * 100.0),
        "mae": float(err.mean()),
        "medae": float(np.median(err)),
    }


def score(truth_d, pred_d, truth_t, pred_t) -> dict:
    out = {
        "distance": {"overall": metrics(truth_d, pred_d), "by_bucket": {}},
        "duration": {"overall": metrics(truth_t, pred_t), "by_bucket": {}},
    }
    # Both targets are bucketed by OSRM *distance* (never by their own units), matching
    # tests/osrm_vs_onnx.py --stratify, so the "<1km" columns mean the same thing in both.
    for key, truth, pred in (("distance", truth_d, pred_d), ("duration", truth_t, pred_t)):
        for label, lo, hi in BUCKETS:
            mask = (truth_d >= lo) & (truth_d < hi)
            if mask.sum() >= 100:
                out[key]["by_bucket"][label] = metrics(truth[mask], pred[mask])
    return out


def train_and_score(X, y_d, y_t, w, X_te, leaves: int) -> tuple[dict, float]:
    t0 = time.time()
    params = {**BASE_PARAMS, "num_leaves": leaves}
    preds = {}
    for name, y in (("distance", y_d), ("duration", y_t)):
        model = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w)
        preds[name] = model.predict(X_te)
    return preds, time.time() - t0


def score_onnx(path: Path, X_te: np.ndarray) -> dict | None:
    """Score a shipped .onnx artifact on the same rows (contaminated — see docstring)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outputs = [o.name for o in sess.get_outputs()]
    arrs = sess.run(None, {inp.name: X_te.astype(np.float32)})

    def pick(*needles: str) -> int:
        for i, name in enumerate(outputs):
            if any(n in name.lower() for n in needles):
                return i
        return -1

    di = pick("distance")
    ti = pick("duration")
    if di < 0 or ti < 0:  # fall back to the documented export order
        di, ti = (0, 1) if "distance" not in outputs[1].lower() else (1, 0)
    return {"distance": np.asarray(arrs[di]).ravel(), "duration": np.asarray(arrs[ti]).ravel()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaves", type=int, nargs="+", default=[63, 511])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--artifacts", action="store_true",
                    help="also score the two shipped .onnx files (contaminated head-to-head)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "scratch" / "capacity_accuracy.json")
    args = ap.parse_args()

    t0 = time.time()
    samples = pd.read_parquet(PROC / "samples.parquet")
    off = pd.read_parquet(PROC / "offnetwork.parquet")

    # Same split as experiments/try_s1_fair.py: seed 0, first fifth held out.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(off))
    n_te = int(len(off) * args.test_frac)
    te = off.iloc[perm[:n_te]].reset_index(drop=True)
    tr_off = off.iloc[perm[n_te:]].reset_index(drop=True)
    print(f"[price] samples={len(samples):,}  off-train={len(tr_off):,}  off-test={len(te):,}")

    truth_d = te["osrm_distance_m"].to_numpy(np.float64)
    truth_t = te["osrm_duration_s"].to_numpy(np.float64)

    both = pd.concat([samples, tr_off], ignore_index=True)
    X = geometric_features(both["orig_lat"].to_numpy(), both["orig_lon"].to_numpy(),
                           both["dest_lat"].to_numpy(), both["dest_lon"].to_numpy())
    w = inverse_frequency_weights(both["osrm_distance_m"].to_numpy(np.float64))
    X_te = geometric_features(te["orig_lat"].to_numpy(), te["orig_lon"].to_numpy(),
                              te["dest_lat"].to_numpy(), te["dest_lon"].to_numpy())
    y_d = both["osrm_distance_m"].to_numpy(np.float32)
    y_t = both["osrm_duration_s"].to_numpy(np.float32)
    print(f"[price] features built ({time.time() - t0:.0f}s); "
          f"train={len(both):,} x {X.shape[1]}")

    results: dict[str, dict] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for leaves in args.leaves:
        print(f"[price] training num_leaves={leaves} ...", flush=True)
        preds, secs = train_and_score(X, y_d, y_t, w, X_te, leaves)
        rec = score(truth_d, preds["distance"], truth_t, preds["duration"])
        rec["leaves"] = leaves
        rec["fit_seconds"] = round(secs, 1)
        results[f"leaves={leaves}"] = rec
        # Checkpoint after every capacity: a long 511-leaf fit should not be able to
        # discard the 63/127/255 rows if it is interrupted.
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        d, u = rec["distance"]["overall"], rec["duration"]["overall"]
        print(f"[price]   leaves={leaves}: dist MedAPE={d['medape']:.1f}% MAE={d['mae']:,.0f} | "
              f"dur MedAPE={u['medape']:.1f}% MAE={u['mae']:,.0f}  ({secs:.0f}s)", flush=True)

    if args.artifacts:
        shipped = {"shipped-511leaf-onnx": ROOT / "server" / "models" / "model.onnx",
                   "old-63leaf-onnx": ROOT / "experiments" / "scratch" / "model_small.onnx"}
        for label, path in shipped.items():
            print(f"[price] scoring artifact {label} ...", flush=True)
            try:
                preds = score_onnx(path, X_te)
            except Exception as exc:  # noqa: BLE001
                print(f"[price]   skipped {label}: {exc}")
                continue
            rec = score(truth_d, preds["distance"], truth_t, preds["duration"])
            rec["artifact_mb"] = round(path.stat().st_size / 1e6, 2)
            rec["contaminated"] = True
            results[label] = rec

    print("\n=== MedAPE by OSRM distance bucket (held-out raw coordinates) ===")
    header = f"{'config':24s} {'target':9s} {'ALL':>8s} " + " ".join(f"{b:>8s}" for b, _, _ in BUCKETS)
    print(header)
    for label, rec in results.items():
        for target in ("distance", "duration"):
            overall = rec[target]["overall"]["medape"]
            cells = " ".join(
                f"{rec[target]['by_bucket'].get(b, {}).get('medape', float('nan')):8.1f}"
                for b, _, _ in BUCKETS
            )
            print(f"{label:24s} {target:9s} {overall:8.1f} {cells}")
    print(f"\n[price] wrote {args.out}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
