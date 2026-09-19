#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "requests>=2.32",
#   "scipy>=1.11",
# ]
# ///
"""Rigorous accuracy statistics: exact OSRM vs the ONNX approximation service.

Unlike tests/benchmark.py (which reports a couple of summary numbers alongside
latency/memory), this script is *only* about the accuracy question and answers
it properly:

  * samples random coordinate pairs across Greater Manchester (optionally
    restricted to points that actually snap to a road, so OSRM and the model are
    compared on the same inputs);
  * obtains exact ground truth from a live OSRM via chunked /table;
  * obtains predictions from the running .NET/ONNX service over HTTP;
  * reports the full error picture -- MAE / RMSE, signed bias with a bootstrap
    confidence interval, the percentile distribution of absolute percentage
    error, correlation and an OLS fit of pred ~ truth -- overall and stratified
    by trip distance.

Outputs a human-readable Markdown report and a JSON sidecar.

This is a self-contained uv script (PEP 723): run it directly, no project
install needed --

    ./tests/osrm_vs_onnx.py --n 5000            # or: uv run tests/osrm_vs_onnx.py
    uv run tests/osrm_vs_onnx.py --n 5000 --stratify
    uv run tests/osrm_vs_onnx.py --n 5000 --on-network-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]

# Reuse the exact OSRM client that produced the training data.
sys.path.insert(0, str(ROOT / "python"))
from fetch_osrm_matrix import OsrmClient  # noqa: E402

# Greater Manchester sample window (inset from the county bbox).
LAT_RANGE = (53.34, 53.64)
LON_RANGE = (-2.72, -1.95)

# Distance buckets for the stratified breakdown (metres), matching the
# separation bands used in train_export_onnx.py.
DISTANCE_BUCKETS = [
    ("<1 km", 0.0, 1_000.0),
    ("1-3 km", 1_000.0, 3_000.0),
    ("3-10 km", 3_000.0, 10_000.0),
    ("10-25 km", 10_000.0, 25_000.0),
    (">25 km", 25_000.0, np.inf),
]


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sample_pairs(n, rng):
    """Random (origin, destination) pairs, rejection-sampling trivially short hops."""
    out = []
    while len(out) < n:
        lat1 = rng.uniform(*LAT_RANGE)
        lon1 = rng.uniform(*LON_RANGE)
        lat2 = rng.uniform(*LAT_RANGE)
        lon2 = rng.uniform(*LON_RANGE)
        if haversine_m(lat1, lon1, lat2, lon2) < 150.0:
            continue
        out.append((lat1, lon1, lat2, lon2))
    return np.asarray(out, dtype=np.float64)


# Separation bands (metres, great-circle) used by --stratify so that short trips
# are not drowned out by the long ones that dominate uniform bbox sampling.
HOP_BANDS = [(200, 1_000), (1_000, 3_000), (3_000, 10_000), (10_000, 25_000), (25_000, 60_000)]


def sample_pairs_stratified(n, rng):
    """Even coverage across separation bands: pick an origin, then a destination
    at a random bearing and a distance sampled from each band."""
    per = max(1, n // len(HOP_BANDS))
    out = []
    for lo, hi in HOP_BANDS:
        got = 0
        while got < per:
            lat1 = rng.uniform(*LAT_RANGE)
            lon1 = rng.uniform(*LON_RANGE)
            heading = rng.uniform(0.0, 2.0 * np.pi)
            d = rng.uniform(lo, hi)
            lat2 = lat1 + (d * np.cos(heading)) / 111_320.0
            lon2 = lon1 + (d * np.sin(heading)) / (111_320.0 * np.cos(np.radians(lat1)))
            if not (LAT_RANGE[0] <= lat2 <= LAT_RANGE[1] and LON_RANGE[0] <= lon2 <= LON_RANGE[1]):
                continue
            out.append((lat1, lon1, lat2, lon2))
            got += 1
    return np.asarray(out, dtype=np.float64)


def osrm_truth(osrm: OsrmClient, pairs, chunk: int = 500):
    """Exact durations/distances for pairs[i] via the diagonal of chunked /table blocks."""
    n = len(pairs)
    dur = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pairs[start:stop]
        d, m = osrm.table(
            block[:, 0], block[:, 1], block[:, 2], block[:, 3]
        )
        idx = np.arange(stop - start)
        dur[start:stop] = d[idx, idx]
        dist[start:stop] = m[idx, idx]
    return dur, dist


def service_predict(service: str, session: requests.Session, pairs):
    dur = np.empty(len(pairs))
    dist = np.empty(len(pairs))
    for i, (lat1, lon1, lat2, lon2) in enumerate(pairs):
        r = session.get(
            f"{service}/route",
            params={"orig": f"{lat1},{lon1}", "dest": f"{lat2},{lon2}"},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        dur[i] = body["duration_s"]
        dist[i] = body["distance_m"]
    return dur, dist


def snap_distances(osrm: OsrmClient, pairs, workers: int = 24):
    """Distance-to-nearest-road for every endpoint, via /nearest (one coord per call)."""
    lats = np.concatenate([pairs[:, 0], pairs[:, 2]])
    lons = np.concatenate([pairs[:, 1], pairs[:, 3]])
    waypoints = osrm.nearest(lats, lons, workers=workers)
    d = np.asarray([w["distance"] for w in waypoints], dtype=np.float64)
    return np.maximum(d[: len(pairs)], d[len(pairs):])


def boot_ci(values, stat, n_resamples=2000, seed=0):
    res = sps.bootstrap(
        (np.asarray(values, dtype=np.float64),),
        stat,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method="percentile",
        random_state=seed,
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high)


def error_stats(truth, pred, seed=0):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    keep = np.isfinite(truth) & np.isfinite(pred) & (truth > 0)
    truth, pred = truth[keep], pred[keep]
    n = len(truth)

    err = pred - truth
    aerr = np.abs(err)
    ape = aerr / truth

    slope, intercept = np.polyfit(truth, pred, 1)
    pearson_r = sps.pearsonr(truth, pred).statistic
    spearman_r = sps.spearmanr(truth, pred).statistic

    mae_lo, mae_hi = boot_ci(aerr, np.mean, seed=seed)
    med_lo, med_hi = boot_ci(ape, np.median, seed=seed)

    return {
        "n": int(n),
        "truth_mean": float(truth.mean()),
        "pred_mean": float(pred.mean()),
        "mae": float(aerr.mean()),
        "mae_ci95": [mae_lo, mae_hi],
        "medae": float(np.median(aerr)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias_mean": float(err.mean()),
        "bias_median": float(np.median(err)),
        "bias_ci95": list(boot_ci(err, np.mean, seed=seed)),
        "bias_pct": float(100.0 * err.mean() / truth.mean()),
        "mape_pct": float(100.0 * ape.mean()),
        "medape_pct": float(100.0 * np.median(ape)),
        "medape_ci95_pct": [100.0 * med_lo, 100.0 * med_hi],
        "ape_pct": {p: float(100.0 * np.percentile(ape, p)) for p in (50, 75, 90, 95, 99)},
        "within": {t: float(100.0 * (ape <= t).mean()) for t in (0.05, 0.10, 0.20, 0.25)},
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
    }


def stratified(bucket_key, truth, pred, seed=0):
    """Break error statistics down by a bucketing variable.

    `bucket_key` decides membership (e.g. OSRM distance); `truth`/`pred` are the
    values actually scored, so units are never mixed.
    """
    out = {}
    bucket_key = np.asarray(bucket_key, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    for name, lo, hi in DISTANCE_BUCKETS:
        mask = (bucket_key >= lo) & (bucket_key < hi) & np.isfinite(bucket_key)
        if mask.sum() < 30:
            continue
        out[name] = error_stats(truth[mask], pred[mask], seed=seed)
    return out


def fmt_table(title, rows):
    lines = [f"#### {title}", "", "| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows.items():
        lines.append(
            f"| {name} | {s['n']:,} | {s['mae']:,.1f} | {s['medae']:,.1f} | {s['rmse']:,.1f} | "
            f"{s['bias_mean']:+,.1f} | {s['medape_pct']:.1f}% | {s['ape_pct'][90]:.1f}% | "
            f"{s['within'][0.10]:.0f}% |"
        )
    lines.append("")
    return lines


def describe(label, s, unit):
    return [
        f"### {label}",
        "",
        f"- **n** = {s['n']:,} pairs (all routable)",
        f"- **MAE** = {s['mae']:,.1f} {unit}  (95% CI {s['mae_ci95'][0]:,.1f} – {s['mae_ci95'][1]:,.1f})",
        f"- **MedAE** = {s['medae']:,.1f} {unit}",
        f"- **RMSE** = {s['rmse']:,.1f} {unit}",
        f"- **MedAPE** = {s['medape_pct']:.1f}%  (95% CI {s['medape_ci95_pct'][0]:.1f} – {s['medape_ci95_pct'][1]:.1f})",
        f"- **MAPE** (mean) = {s['mape_pct']:.1f}%",
        f"- **signed bias** = {s['bias_mean']:+,.1f} {unit} ({s['bias_pct']:+.1f}% of mean truth; "
        f"95% CI {s['bias_ci95'][0]:+,.1f} – {s['bias_ci95'][1]:+,.1f})",
        f"- **APE percentiles** = p50 {s['ape_pct'][50]:.1f}% · p75 {s['ape_pct'][75]:.1f}% · "
        f"p90 {s['ape_pct'][90]:.1f}% · p95 {s['ape_pct'][95]:.1f}% · p99 {s['ape_pct'][99]:.1f}%",
        f"- **within** 5% {s['within'][0.05]:.0f}% · 10% {s['within'][0.10]:.0f}% · "
        f"20% {s['within'][0.20]:.0f}% · 25% {s['within'][0.25]:.0f}%",
        f"- **correlation** Pearson r = {s['pearson_r']:.4f}, Spearman ρ = {s['spearman_r']:.4f}",
        f"- **OLS fit** pred ≈ {s['ols_intercept']:.1f} + {s['ols_slope']:.4f}·truth "
        f"({'(slight under-prediction)' if s['bias_mean'] < 0 else '(slight over-prediction)'})",
        "",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service", default="http://localhost:5080")
    ap.add_argument("--osrm", default="http://localhost:5001")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--on-network-only", action="store_true",
                    help="drop pairs whose endpoints are far from any road (apples-to-apples)")
    ap.add_argument("--stratify", action="store_true",
                    help="sample evenly across separation bands instead of uniform bbox sampling")
    ap.add_argument("--max-snap-m", type=float, default=250.0)
    ap.add_argument("--out", default=str(ROOT / "tests" / "osrm_vs_onnx_report.json"))
    ap.add_argument("--md", default=str(ROOT / "tests" / "OSRM_VS_ONNX.md"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    osrm = OsrmClient(args.osrm)

    # health
    health = session.get(f"{args.service}/health", timeout=10).json()
    print(f"[stats] service model={health.get('model')}")

    pairs = sample_pairs_stratified(args.n, rng) if args.stratify else sample_pairs(args.n, rng)
    print(f"[stats] sampled {len(pairs):,} pairs (seed={args.seed}, "
          f"{'stratified' if args.stratify else 'uniform'})")

    dropped_snap = 0
    if args.on_network_only:
        snap = snap_distances(osrm, pairs)
        keep = snap <= args.max_snap_m
        dropped_snap = int((~keep).sum())
        pairs = pairs[keep]
        print(f"[stats] on-network filter: kept {len(pairs):,}, dropped {dropped_snap:,} "
              f"(> {args.max_snap_m:.0f} m from a road)")

    print(f"[stats] OSRM /table ground truth for {len(pairs):,} pairs ...")
    truth_dur, truth_dist = osrm_truth(osrm, pairs)

    routable = np.isfinite(truth_dur) & np.isfinite(truth_dist) & (truth_dist > 0)
    dropped_route = int((~routable).sum())
    pairs, truth_dur, truth_dist = pairs[routable], truth_dur[routable], truth_dist[routable]
    print(f"[stats] routable pairs: {len(pairs):,} (dropped {dropped_route:,} unroutable)")

    print(f"[stats] querying ONNX service for {len(pairs):,} pairs ...")
    pred_dur, pred_dist = service_predict(args.service, session, pairs)

    dur = error_stats(truth_dur, pred_dur, seed=args.seed)
    dist = error_stats(truth_dist, pred_dist, seed=args.seed)
    dur_by = stratified(truth_dist, truth_dur, pred_dur, seed=args.seed)
    dist_by = stratified(truth_dist, truth_dist, pred_dist, seed=args.seed)

    generated = datetime.now(timezone.utc).isoformat()
    md = [
        "# OSRM vs ONNX approximation — accuracy statistics",
        "",
        f"_Generated {generated} by `tests/osrm_vs_onnx.py`._",
        "",
        f"- Sample: **{len(pairs):,}** {'stratified-across-bands' if args.stratify else 'uniformly random'} pairs (seed {args.seed})"
        + (f", restricted to points within {args.max_snap_m:.0f} m of a road"
           f" (dropped {dropped_snap:,} off-network)" if args.on_network_only else ""),
        f"- Ground truth: live OSRM `/table` · Predictions: `GET /route` on {args.service}",
        "- Units: duration in seconds, distance in metres. APE = |pred − truth| / truth.",
        "",
        "## Headline",
        "",
        "| metric | duration | distance |",
        "|---|---|---|",
        f"| MAE | {dur['mae']:,.1f} s | {dist['mae']:,.1f} m |",
        f"| MedAE | {dur['medae']:,.1f} s | {dist['medae']:,.1f} m |",
        f"| RMSE | {dur['rmse']:,.1f} s | {dist['rmse']:,.1f} m |",
        f"| MedAPE | {dur['medape_pct']:.1f}% | {dist['medape_pct']:.1f}% |",
        f"| MAPE | {dur['mape_pct']:.1f}% | {dist['mape_pct']:.1f}% |",
        f"| within 10% | {dur['within'][0.10]:.0f}% | {dist['within'][0.10]:.0f}% |",
        f"| within 25% | {dur['within'][0.25]:.0f}% | {dist['within'][0.25]:.0f}% |",
        f"| Pearson r | {dur['pearson_r']:.4f} | {dist['pearson_r']:.4f} |",
        f"| bias | {dur['bias_mean']:+,.1f} s | {dist['bias_mean']:+,.1f} m |",
        "",
        "## Detail",
        "",
    ]
    md += describe("Duration", dur, "s")
    md += describe("Distance", dist, "m")
    md += ["## Stratified by OSRM distance", ""]
    md += fmt_table("Duration", dur_by)
    md += fmt_table("Distance", dist_by)

    report = {
        "generated_at": generated,
        "service": args.service,
        "osrm": args.osrm,
        "seed": args.seed,
        "sampling": "stratified" if args.stratify else "uniform",
        "requested_n": args.n,
        "used_n": int(len(pairs)),
        "dropped_unroutable": dropped_route,
        "dropped_off_network": dropped_snap,
        "on_network_only": bool(args.on_network_only),
        "duration": dur,
        "distance": dist,
        "duration_by_distance": dur_by,
        "distance_by_distance": dist_by,
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[stats] wrote {args.out}")
    print(f"[stats] wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
