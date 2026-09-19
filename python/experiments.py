#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=17.0",
#   "lightgbm>=4.5",
#   "scikit-learn>=1.5",
# ]
# ///
"""Offline experiment harness for model improvements (see IMPROVEMENTS.md).

Trains a set of variants on data/processed/samples.parquet and scores them on a
*fixed, distance-balanced* held-out set, so the numbers are comparable across
experiments and not skewed by the all-pairs distribution (which is 99.8% > 1 km).

    ./python/experiments.py                 # all variants below
    ./python/experiments.py --train-rows 2000000
    ./python/experiments.py --only E1_invfreq E2_log

Writes experiments/results.json and experiments/results.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "data" / "processed" / "samples.parquet"

# Buckets by OSRM road distance (metres), same bands used everywhere else.
BUCKETS = [
    ("<1km", 0.0, 1_000.0),
    ("1-3km", 1_000.0, 3_000.0),
    ("3-10km", 3_000.0, 10_000.0),
    ("10-25km", 10_000.0, 25_000.0),
    (">25km", 25_000.0, float("inf")),
]

BASE_FEATURES = [
    "orig_lat", "orig_lon", "dest_lat", "dest_lon",
    "haversine_dist_m", "bearing_deg", "lat_delta", "lon_delta",
]

DENSITY_PATH = ROOT / "data" / "processed" / "road_density.npz"
_density_cache: dict = {}


def load_density():
    """E4: lazily load the road-density raster built by build_road_density.py."""
    if "arr" not in _density_cache:
        import json as _json
        data = np.load(DENSITY_PATH, allow_pickle=False)
        meta = _json.loads(str(data["meta"]))
        _density_cache["arr"] = data["density"]
        _density_cache["meta"] = meta
    return _density_cache["arr"], _density_cache["meta"]


def road_density(lat, lon):
    """Total drivable-road metres in the ~250 m cell containing each point."""
    arr, meta = load_density()
    ix = np.floor((np.asarray(lon) - meta["min_lon"]) / meta["cell"]).astype(np.int64)
    iy = np.floor((np.asarray(lat) - meta["min_lat"]) / meta["cell"]).astype(np.int64)
    inside = (ix >= 0) & (ix < meta["nx"]) & (iy >= 0) & (iy < meta["ny"])
    out = np.zeros(len(ix), dtype=np.float64)
    out[inside] = arr[iy[inside], ix[inside]]
    return out



def haversine(lat1, lon1, lat2, lon2):
    r = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def build_features(df, kind):
    lat1 = df["orig_lat"].to_numpy(np.float64)
    lon1 = df["orig_lon"].to_numpy(np.float64)
    lat2 = df["dest_lat"].to_numpy(np.float64)
    lon2 = df["dest_lon"].to_numpy(np.float64)
    hav = haversine(lat1, lon1, lat2, lon2)
    brg = bearing(lat1, lon1, lat2, lon2)
    base = np.column_stack([lat1, lon1, lat2, lon2, hav, brg, lat2 - lat1, lon2 - lon1]).astype(np.float32)
    if kind == "base":
        return base, hav
    if kind == "sincos":
        # Replace bearing_deg with sin/cos: axis-aligned splits can't handle the
        # 0/360 wrap, but sin/cos is continuous around the compass.
        out = np.column_stack([
            lat1, lon1, lat2, lon2, hav,
            np.sin(np.radians(brg)), np.cos(np.radians(brg)),
            lat2 - lat1, lon2 - lon1,
        ]).astype(np.float32)
        return out, hav
    if kind == "density":
        # E4: append log1p road density at both endpoints.
        od = np.log1p(road_density(lat1, lon1))
        dd = np.log1p(road_density(lat2, lon2))
        out = np.column_stack([lat1, lon1, lat2, lon2, hav, brg, lat2 - lat1, lon2 - lon1, od, dd]).astype(np.float32)
        return out, hav
    raise ValueError(f"unknown feature set {kind}")


def bucket_labels(dist, values):
    labels = np.empty(len(values), dtype=object)
    labels[:] = "?"
    for name, lo, hi in BUCKETS:
        labels[(values >= lo) & (values < hi)] = name
    return labels


def encode_target(df, hav, target, target_name):
    """Return (y, inverse) where inverse maps model output back to real units."""
    raw = df[target_name].to_numpy(np.float64)
    if target == "raw":
        return raw, (lambda p: p)
    if target == "log":
        return np.log(raw), (lambda p: np.exp(p))
    if target == "norm":
        if target_name == "osrm_distance_m":
            return raw / hav, (lambda p: p * hav)
        return raw / hav, (lambda p: p * hav)  # pace (s per metre of straight line)
    raise ValueError(f"unknown target {target}")


def train_params(args):
    return {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.estimators,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "verbose": -1,
        "num_threads": args.threads,
    }


def make_weights(train_scores, scheme):
    if scheme == "none":
        return None
    if scheme == "inv_freq":
        labels = bucket_labels(None, train_scores)
        w = np.ones(len(train_scores), dtype=np.float64)
        for name, _, _ in BUCKETS:
            m = labels == name
            if m.any():
                w[m] = 1.0 / m.sum()
        w *= len(w) / w.sum()  # normalise to mean 1
        return w
    raise ValueError(scheme)


def medape(y_true, y_pred):
    rel = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)
    return float(np.median(rel) * 100.0)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(predict_fn, X_te, hav_te, df_te, cfg):
    out = {}
    for target_name, key in (("distance_m", "osrm_distance_m"), ("duration_s", "osrm_duration_s")):
        _, inverse = encode_target(df_te, hav_te, cfg["target"], key)
        pred = inverse(predict_fn(key, X_te, hav_te))
        truth = df_te[key].to_numpy(np.float64)
        labels = bucket_labels(None, truth)
        rec = {"overall": {"medape": medape(truth, pred), "mae": mae(truth, pred), "n": int(len(truth))}}
        by = {}
        for name, _, _ in BUCKETS:
            m = labels == name
            if m.sum() < 20:
                continue
            by[name] = {"medape": medape(truth[m], pred[m]), "mae": mae(truth[m], pred[m]), "n": int(m.sum())}
        rec["by_bucket"] = by
        out[key] = rec
    return out


VARIANTS = [
    ("baseline", {"features": "base", "target": "raw", "weights": "none"}),
    ("E1_invfreq", {"features": "base", "target": "raw", "weights": "inv_freq"}),
    ("E2_log", {"features": "base", "target": "log", "weights": "none"}),
    ("E2_norm", {"features": "base", "target": "norm", "weights": "none"}),
    ("E3_sincos", {"features": "sincos", "target": "raw", "weights": "none"}),
    ("E1+E2_norm", {"features": "base", "target": "norm", "weights": "inv_freq"}),
    ("E1+E2+E3", {"features": "sincos", "target": "norm", "weights": "inv_freq"}),
    # E7: objective / capacity, each on top of the E1 winner.
    ("E7_huber", {"features": "base", "target": "raw", "weights": "inv_freq",
                  "params": {"objective": "huber"}}),
    ("E7_leaves127", {"features": "base", "target": "raw", "weights": "inv_freq",
                      "params": {"num_leaves": 127}}),
    ("E7_minchild10", {"features": "base", "target": "raw", "weights": "inv_freq",
                       "params": {"min_child_samples": 10}}),
    # E9: tree pruning (accuracy vs size).
    ("E9_trees150", {"features": "base", "target": "raw", "weights": "inv_freq",
                     "params": {"n_estimators": 150}}),
    ("E9_trees60", {"features": "base", "target": "raw", "weights": "inv_freq",
                    "params": {"n_estimators": 60}}),
    # E6: short-range specialist, routed by straight-line separation.
    ("E6_specialist", {"features": "base", "target": "raw", "weights": "inv_freq",
                       "mode": "specialist", "gate_m": 3000.0}),
    # Combinations of the winners.
    ("E7_leaves255", {"features": "base", "target": "raw", "weights": "inv_freq",
                      "params": {"num_leaves": 255}}),
    ("E7_leaves511", {"features": "base", "target": "raw", "weights": "inv_freq",
                      "params": {"num_leaves": 511}}),
    ("E6+E7_l127", {"features": "base", "target": "raw", "weights": "inv_freq",
                    "mode": "specialist", "gate_m": 3000.0, "params": {"num_leaves": 127}}),
    # E4: static road-density features (built by build_road_density.py).
    ("E4_density", {"features": "density", "target": "raw", "weights": "inv_freq"}),
    ("E4_density_l511", {"features": "density", "target": "raw", "weights": "inv_freq",
                         "params": {"num_leaves": 511}}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--test-per-bucket", type=int, default=4000)
    ap.add_argument("--train-rows", type=int, default=2_000_000,
                    help="cap on training rows (0 = use all). Keeps the loop fast.")
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results.json"))
    ap.add_argument("--md", default=str(ROOT / "experiments" / "results.md"))
    args = ap.parse_args()

    print(f"[exp] loading {args.samples}")
    df = pd.read_parquet(args.samples)
    n = len(df)
    print(f"[exp] {n:,} pairs")

    # ---- fixed, distance-balanced test split --------------------------------------
    rng = np.random.default_rng(args.seed)
    dist = df["osrm_distance_m"].to_numpy(np.float64)
    labels = bucket_labels(None, dist)
    test_idx = []
    for name, _, _ in BUCKETS:
        idx = np.flatnonzero(labels == name)
        take = min(args.test_per_bucket, len(idx))
        test_idx.append(rng.choice(idx, size=take, replace=False))
    test_idx = np.concatenate(test_idx)
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.flatnonzero(mask)
    if args.train_rows and len(train_idx) > args.train_rows:
        train_idx = rng.choice(train_idx, size=args.train_rows, replace=False)
    print(f"[exp] test={len(test_idx):,} (balanced)  train={len(train_idx):,}")

    df_tr, df_te = df.iloc[train_idx], df.iloc[test_idx]
    params = train_params(args)

    # Cache features per (kind, split); train and test must not collide.
    feat_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def feats(kind, d, role):
        key = (kind, role)
        if key not in feat_cache:
            feat_cache[key] = build_features(d, kind)
        return feat_cache[key]

    results = {}
    selected = set(args.only) if args.only else None

    for name, cfg in VARIANTS:
        if selected and name not in selected:
            continue
        print(f"\n[exp] === {name} :: {cfg} ===")
        X_tr, hav_tr = feats(cfg["features"], df_tr, "train")
        X_te, hav_te = feats(cfg["features"], df_te, "test")
        scores = df_tr["osrm_distance_m"].to_numpy(np.float64)
        w = make_weights(scores, cfg["weights"])
        p = dict(params)
        p.update(cfg.get("params", {}))
        t0 = time.time()

        if cfg.get("mode", "single") == "specialist":
            # Two boosters per target, routed by straight-line separation.
            gate = cfg.get("gate_m", 3000.0)
            short = hav_tr < gate
            fitted: dict[str, tuple] = {}
            for target_name in ("osrm_distance_m", "osrm_duration_s"):
                y, _ = encode_target(df_tr, hav_tr, cfg["target"], target_name)
                b_short = lgb.LGBMRegressor(**p).fit(
                    X_tr[short], y[short], sample_weight=None if w is None else w[short]).booster_
                b_long = lgb.LGBMRegressor(**p).fit(
                    X_tr[~short], y[~short], sample_weight=None if w is None else w[~short]).booster_
                fitted[target_name] = (b_short, b_long)

            def predict_fn(key, X, hav):
                b_short, b_long = fitted[key]
                out = np.empty(len(X), dtype=np.float64)
                s = hav < gate
                out[s] = b_short.predict(X[s])
                out[~s] = b_long.predict(X[~s])
                return out
        else:
            models = {}
            for target_name in ("osrm_distance_m", "osrm_duration_s"):
                y, _ = encode_target(df_tr, hav_tr, cfg["target"], target_name)
                m = lgb.LGBMRegressor(**p)
                m.fit(X_tr, y, sample_weight=w)
                models[target_name] = m.booster_

            def predict_fn(key, X, hav, _models=models):
                return _models[key].predict(X)

        elapsed = time.time() - t0
        rec = evaluate(predict_fn, X_te, hav_te, df_te, cfg)
        rec["cfg"] = cfg
        rec["fit_seconds"] = round(elapsed, 1)
        results[name] = rec
        d = rec["osrm_distance_m"]
        u = rec["osrm_duration_s"]
        print(f"[exp]   distance MedAPE={d['overall']['medape']:.1f}% MAE={d['overall']['mae']:,.0f}  "
              f"duration MedAPE={u['overall']['medape']:.1f}% MAE={u['overall']['mae']:,.0f}  ({elapsed:.0f}s)")

    payload = {
        "samples": args.samples,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "params": params,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    # ---- markdown table ------------------------------------------------------------
    lines = ["# Experiment results", "",
             f"train={len(train_idx):,} · balanced test={len(test_idx):,} · "
             f"leaves={args.num_leaves} trees={args.estimators} lr={args.learning_rate}", "",
             "| variant | features | target | weights | dist MedAPE | dur MedAPE | "
             "<1km | 1-3km | 3-10km | 10-25km | >25km | fit s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rec in results.items():
        d = rec["osrm_distance_m"]["by_bucket"]
        c = rec["cfg"]
        cells = " | ".join(f"{d.get(b, {}).get('medape', float('nan')):.0f}%" for b in
                           ["<1km", "1-3km", "3-10km", "10-25km", ">25km"])
        lines.append(
            f"| {name} | {c['features']} | {c['target']} | {c['weights']} | "
            f"{rec['osrm_distance_m']['overall']['medape']:.1f}% | "
            f"{rec['osrm_duration_s']['overall']['medape']:.1f}% | {cells} | {rec['fit_seconds']:.0f} |"
        )
    lines.append("")
    lines.append("_Distance MedAPE per bucket (<1km … >25km); all values on the same balanced held-out set._")
    Path(args.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[exp] wrote {args.out}")
    print(f"[exp] wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
