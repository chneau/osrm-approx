# Model improvement roadmap

Goal: improve the shipped ONNX model (`server/models/model.onnx`), one change at a
time, recording the measured effect of each **even when it makes things worse** (a
documented negative result is a result).

**Status: complete.** Every experiment (E1–E9) has been run and recorded below. Two
changes were adopted (E1, E5) plus a capacity bump (E7); the model was retrained,
re-exported and re-verified end to end. Four changes were rejected on measurement,
and one (E8) turned out to be a non-issue. The export pipeline also gained a real
correctness fix (float32 threshold rounding) found while shipping a larger model.

## Evidence: why the model originally failed at short range

Two independent measurements point at the same root cause.

**1. The training pairs are overwhelmingly long trips.** `data/processed/samples.parquet`
is the all-pairs set of 2,203 routable grid points (4,850,998 pairs):

| separation (OSRM road distance) | share of training pairs |
|---|---|
| < 1 km | 0.23% |
| < 3 km | 2.6% |
| > 25 km | 29.6% |
| median | 17.3 km (p1 = 1.9 km) |

The objective is MAE over that distribution, so the model was optimised for long
trips and barely saw short ones.

**2. Accuracy was strongly distance-dependent** (band-balanced service eval of the
*original* 63-leaf model, `tests/osrm_vs_onnx.py --stratify`, N=4,682):

| OSRM distance | duration MedAPE | distance MedAPE | distance bias |
|---|---|---|---|
| < 1 km | 176.5% | 216.9% | +1,962 m |
| 1–3 km | 34.5% | 38.5% | +1,319 m |
| 3–10 km | 20.4% | 21.0% | +386 m |
| 10–25 km | 9.9% | 11.4% | +550 m |
| > 25 km | 5.6% | 5.0% | +213 m |

The model had an implicit ~1 km floor and could not represent a sub-kilometre road
trip. A smooth coordinate function also cannot express network discontinuities, which
sets the ceiling on feature engineering alone.

## Protocol

- **Harness**: `python/experiments.py` trains a variant and evaluates it on a
  *fixed, distance-balanced held-out set* (frozen seed) so every experiment is
  comparable. It reports MAE / MedAE / MedAPE overall and per distance bucket.
- **Baseline** is the original shipped configuration (63 leaves, no sample weights)
  run through the same harness. The numbers in `server/models/model_metadata.json`
  come from a random split of the all-pairs set, which is long-trip dominated and
  not the right gate.
- **Primary gate**: MedAPE per distance bucket on the balanced held-out set.
  A change is an improvement only if the short/mid buckets get better without a
  material regression above 10 km.
- **End-to-end confirmation**: the adopted changes were retrained into the shipped
  model, served, and re-measured with `./tests/osrm_vs_onnx.py` and
  `tests/benchmark.py`.
- Every experiment records its result below, good or bad.

## Experiments

| id | change | hypothesis | status |
|----|--------|-----------|--------|
| E1 | Rebalance training separations (inverse-frequency sample weights) | Short trips are 0.2% of data; up-weighting them removes the 1 km floor | **done — big win, adopted** |
| E2 | Log / normalised targets (`log distance`, detour factor, pace) | Aligns the loss with relative error | **done — partial, rejected vs E1** |
| E3 | `sin`/`cos(bearing)` instead of `bearing_deg` | Removes the 0/360 wrap discontinuity | **done — null result, rejected** |
| E4 | Static road-density features at origin & dest | Encodes urban-vs-rural speed, information coordinates don't carry | **done — null result, rejected** |
| E5 | Include off-network points in training | Matches inference, which receives arbitrary raw coordinates | **done — win on raw coords, adopted** |
| E6 | Short-range specialist model (<3 km), gated by haversine | Concentrates capacity where the error is | **done — rejected** |
| E7 | Objective / capacity tuning (`huber`, leaves, `min_child_samples`) | Resolution for the short regime | **done — capacity (511 leaves) adopted** |
| E8 | ONNX Runtime session tuning (mem arena) | Cut the RSS | **done — null result (already tuned)** |
| E9 | Tree pruning (400 → N trees) | Size/latency vs accuracy trade | **done — pruning costs accuracy** |

### Notes on each

**E1.** The all-pairs set already contains short pairs, just rarely, so no new OSRM
data was needed: sample weights ∝ 1/bucket-frequency. Risk: over-fitting the small
pool of unique short pairs (only ~11k < 1 km). A denser core grid is the real
long-term fix. Implemented as `inverse_frequency_weights()` in
`train_export_onnx.py` (`--weighting inv_freq`, now the default).

**E2.** Train on `log(road_distance)`, `road_distance/haversine` (detour ≥ 1),
`duration/haversine` (pace), reconstruct at inference. Helps alone but is dominated
by E1 and hurts when stacked with it, so it is not shipped.

**E3.** Feature-contract change (would need mirroring in `server/Program.cs`). No
measurable effect, so the contract was left alone.

**E4.** Implemented for real: `python/build_road_density.py` parses the OSM extract
with `pyosmium` and rasterises total drivable-road metres per ~250 m cell
(`data/processed/road_density.npz`). `experiments.py` gained a `density` feature kind
that appends `log1p` density at both endpoints. Despite the extra signal it changed
nothing (see log), so it is not shipped — which is itself informative: the model is
not density-limited, it is starved of *short examples* (E1) and *raw-coordinate*
examples (E5).

**E5.** `python/gen_offnetwork.py` samples random coordinates in the bbox and labels
them with live OSRM `/table` (OSRM snaps them, exactly as the service is handed raw
coords). Adopted: `train_export_onnx.py --extra-samples` auto-loads
`data/processed/offnetwork.parquet` when present.

**E6.** Two boosters per target, routed by haversine. More moving parts, no accuracy
gain over the single larger model — rejected.

**E7/E9.** Capacity keeps helping (63 → 511 leaves); pruning (fewer trees) always
costs accuracy. The cost of 511 leaves is model size and RSS (see below), and it
exposed a real ONNX export precision bug.

**E8.** Already had `InterOpNumThreads=1`, `IntraOpNumThreads=1`,
`ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`. Only the CPU memory arena was untested; disabling
it changed RSS by ~1 MB (noise). The RSS is driven by model size, not the arena.

## Export correctness fix (found while shipping E7)

Shipping the 511-leaf model made the ONNX-vs-LightGBM verification fail: only
**99.24%** of rows agreed within 1%/1 unit, with individual errors up to **6,157 m**.
Root cause: ONNX `TreeEnsembleRegressor` stores split thresholds as `float32` while
LightGBM compares in `float64`, and rounding the threshold *to nearest* can round it
*up* past the true threshold, sending an input between the two values down the wrong
branch. With shallow 63-leaf trees that never bit (leaf values are close); with
511-leaf trees a wrong branch can move the prediction by kilometres.

Fix: round every threshold **down** to the largest `float32 ≤ threshold`
(`floor_float32()` in `train_export_onnx.py`). For `x <= thr` this makes the float32
ONNX comparison exactly equivalent to LightGBM's float64 comparison for any float32
input. After the fix the export matches to **100.0000%** (max |Δ| = 0.03 m for
distance, 0.002 s for duration). This is a strict correctness win independent of
model size.

## Results log

All rows below come from one harness configuration
(`python/experiments.py --train-rows 2000000 --test-per-bucket 4000`, 16 threads):
2M training rows, a fixed distance-balanced 20k-pair held-out set, 63 leaves and 400
trees unless the row overrides them. `dist`/`dur` are overall MedAPE; the bucket
columns are distance MedAPE (`<1km … >25km`).

| id | change | dist MedAPE | dur MedAPE | <1 km | 1-3 km | 3-10 km | 10-25 km | >25 km | verdict |
|----|--------|-------------|------------|-------|--------|---------|----------|--------|---------|
| — | baseline (original shipped config) | 10.3% | 7.3% | 84% | 16% | 9% | 7% | 3% | — |
| E1 | inverse-frequency weights | **7.9%** | **6.4%** | **16%** | **12%** | 8% | 8% | 3% | **win — adopted** |
| E2 | log target | 9.0% | 6.7% | 39% | 13% | 7% | 7% | 4% | partial |
| E2 | normalised target (detour/pace) | 9.6% | 7.5% | 41% | 14% | 8% | 8% | 4% | marginal |
| E3 | sin/cos bearing | 10.2% | 7.2% | 82% | 16% | 8% | 7% | 3% | **no effect — reject** |
| E1+E2 | inv_freq + normalised target | 8.8% | 8.3% | 19% | 10% | 8% | 10% | 5% | worse than E1 alone |
| E1+E2+E3 | all three | 9.0% | 8.5% | 20% | 10% | 8% | 10% | 5% | worse than E1 alone |
| E6 | short-range specialist | 7.4% | 5.9% | 18% | 9% | 7% | 8% | 3% | beaten by E7 capacity |
| E6+E7 | specialist + 127 leaves | 6.6% | 5.1% | 19% | 8% | 7% | 7% | 2% | worse than 255/511 single |
| E7 | `huber` objective | **75.7%** | 58.2% | — | — | — | — | — | **genuine negative** |
| E7 | `min_child_samples=10` | 7.9% | 6.4% | 16% | 12% | 8% | 8% | 3% | no change |
| E7 | 127 leaves | 6.9% | 5.4% | 16% | 11% | 7% | 6% | 2% | better |
| E7 | 255 leaves | 6.0% | 4.4% | 16% | 10% | 6% | 5% | 2% | better |
| E7 | **511 leaves** | **5.1%** | **3.7%** | 16% | 9% | 5% | 4% | 2% | **win — adopted** |
| E9 | 150 trees | 9.0% | 7.8% | 17% | 12% | 8% | 10% | 4% | pruning hurts |
| E9 | 60 trees | 9.9% | 9.1% | 21% | 13% | 9% | 11% | 5% | pruning hurts more |
| E4 | road-density features | 7.8% | 6.3% | 16% | 12% | 8% | 8% | 3% | **null — reject** |
| E4 | road-density features, 511 leaves | 5.1% | 3.6% | 17% | 9% | 5% | 4% | 2% | null vs 511 leaves |

### What the harness showed

- **E1 (rebalance) is the clear winner.** Overall distance MedAPE 10.3% → 7.9%,
  duration 7.3% → 6.4%, and the `< 1 km` bucket collapses **84% → 16%** — the ~1 km
  floor is gone. Cost: the 10–25 km bucket drifts 7% → 8%.
- **E2** helps by itself but is dominated by E1, and **E1+E2 is worse than E1 alone**
  (duration regresses 6.4% → 8.3%). Rejected.
- **E3 is a genuine negative result**: sin/cos bearing changed nothing (10.3% → 10.2%).
- **E4 is a genuine negative result**: real road-density features changed nothing
  (7.9% → 7.8% at 63 leaves; identical at 511). The bottleneck is not feature signal.
- **E6 is rejected**: a haversine-gated short-range specialist (7.4%) does not beat a
  single larger model (255/511 leaves).
- **E7**: capacity is the dominant knob after E1. `huber` is catastrophic
  (75.7%) — robustness to outliers is exactly wrong when the target *is* the
  short-trip tail we just up-weighted. `min_child_samples=10` is neutral.
- **E9**: pruning to 150 or 60 trees always costs accuracy; there is no free
  size/latency win by dropping trees.

### E5 — off-network training data (separate harness)

`python/experiment_e5.py` (127 leaves, 1.5M grid training rows, E1 weights), scoring
two models on two held-out sets. The off-network test set is the API's real use case
(raw coordinates that OSRM must snap).

| model | grid test dist/dur | off-network test dist/dur |
|---|---|---|
| A — grid only | 7.7% / 6.3% | 7.8% / 8.6% |
| B — grid + off-network | 8.0% / 6.6% | **4.5% / 4.3%** |

Off-network breakdown (distance MedAPE, `<1km … >25km`):

| model | <1 km | 1-3 km | 3-10 km | 10-25 km | >25 km |
|---|---|---|---|---|---|
| A grid only | 522% | 144% | 24% | 14% | 6% |
| B grid + off-network | 244% | 36% | 18% | 10% | 3% |

**Adopted.** The grid test is essentially unchanged (7.7→8.0% is within noise), while
accuracy on the raw coordinates the API actually receives roughly **halves**. The
service is handed raw coordinates, so this is the number that matters.

## Shipped configuration (after E1 + E5 + E7)

`python/train_export_onnx.py` now defaults to:

- `--weighting inv_freq` (E1) and `--num-leaves 511` (E7);
- `--extra-samples` auto-loads `data/processed/offnetwork.parquet` when present (E5);
- thresholds floored to float32 on export (correctness fix).

Retrained on the full all-pairs set + 158,806 off-network pairs; 10% held out.

| target | MAE | MedAE | MedAPE | RMSE | ONNX-vs-LightGBM |
|---|---|---|---|---|---|
| `distance_m` | 788.4 | 525.3 | **3.3%** | 1,202.7 | 100.0000% (max Δ 0.03 m) |
| `duration_s` | 36.5 | 29.1 | **2.3%** | 48.9 | 100.0000% (max Δ 0.002 s) |

End-to-end service measurements (live OSRM):

| measurement | original model (63 leaves) | shipped model (511 leaves) |
|---|---|---|
| benchmark MedAPE, 600 off-grid pairs (duration / distance) | 6.6% / 6.7% | **4.9% / 4.1%** |
| band-balanced MedAPE (duration / distance, N≈4.4k) | 12.7% / 14.2% | **9.9% / 10.3%** |
| `< 1 km` distance MedAPE (band-balanced) | 216.9% | **74.4%** |
| `< 1 km` duration MedAPE (band-balanced) | 176.5% | **78.0%** |
| model file | 3.76 MB | 32.44 MB |
| server RSS (warm) | ~138 MB | ~394–430 MB |
| server-side latency p50 / p99 | 104 µs / 240 µs | 296 µs / 842 µs |
| `wrk` throughput (single route) | 44,545 req/s | 43,499 req/s |

### Honest costs of the adopted configuration

- **RSS grows ~3× (138 → ~400 MB)** and the model is **~8.6× larger (3.8 → 32 MB)**.
  Measured directly by loading both models: the jump is the flattened tree node table,
  not the ONNX arena (E8). The plan's < 30 MB RSS target was already missed at 162 MB;
  this makes it worse. It is a deliberate accuracy/size trade — the RSS ladder is
  ~138 MB (63 leaves) → ~400 MB (511 leaves).
- **Server-side p99 rises 240 µs → 842 µs.** It still meets the plan's **p99 < 1 ms**
  target, but the headroom shrinks from ~4× to ~1.2×. Throughput is flat.
- **Short raw-coordinate trips are still the weak point** (`<1 km` ~75% MedAPE): random
  coordinates often snap hundreds of metres, and that snap distance dominates a
  sub-kilometre trip. E1 removed the *grid* floor (84% → 16%); the *raw-coordinate*
  floor is a snapping effect no coordinate model can remove.
- **Long trips stay excellent** (band-balanced `>25 km` ~3.5–4.2%).

## Reproducing

```bash
cd python
uv run experiments.py --train-rows 2000000 --test-per-bucket 4000 --threads 16   # E1-E4,E6,E7,E9
./experiment_e5.py --leaves 127 --estimators 200 --train-rows 1500000            # E5
uv run --with osmium build_road_density.py                                       # E4 raster
uv run gen_offnetwork.py --points 400                                            # E5 data
uv run train_export_onnx.py --threads 16                                         # ship
```
