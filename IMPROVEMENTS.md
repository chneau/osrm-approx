# Model improvement roadmap

Goal: improve the shipped ONNX model (`server/models/model.onnx`), one change at a
time, recording the measured effect of each **even when it makes things worse** (a
documented negative result is a result).

**Status: complete.** Every experiment (E1–E11) has been run and recorded below. Three
changes were adopted (E1, E5, E11) plus a capacity bump (E7); the model was retrained,
re-exported and re-verified end to end. Five changes were rejected on measurement, and
one (E8) turned out to be a non-issue. The export pipeline also gained a real
correctness fix (float32 threshold rounding) found while shipping a larger model. **E11
compiles the trees to C# and drops ONNX Runtime entirely — the first change that moves
server RSS (428 → 136 MB).**

Two follow-up experiments (**Experiment 0**, the exact OSRM snap; **Experiment 1**,
static connectivity/detour rasters) then attacked the raw-coordinate short-trip error
directly. Both are documented below and both were **rejected**: together they move the
off-network short-trip MedAPE from ~39% to ~34% and stop, so the residual is genuine
topology, not snap or barrier geometry. The shipped model is unchanged by them.

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
| E10 | Quantise the ONNX trees (`int8`/`float16`) | Shrink the 32 MB model → RSS | **done — impossible with stock ORT, reverted** |
| E11 | Compile the trees to C# (drop ONNX Runtime) | The only lever that moves RSS | **done — big win, adopted** |
| Exp 0 | Oracle snap (exact OSRM `/nearest` per endpoint) | Snapping is the short-trip bottleneck | **done — real but small, rejected** |
| Exp 1 | Static connectivity/detour rasters | The residual is barrier-forced detour | **done — marginal, rejected** |

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
costs accuracy. That capacity is not free — the "E7 — the capacity price" subsection
below measures it at **+29.4 MB RSS for ~1.8× lower MedAPE** — and shipping a larger
model exposed a real ONNX export precision bug.

**E8.** Already had `InterOpNumThreads=1`, `IntraOpNumThreads=1`,
`ORT_ENABLE_ALL`, `ORT_SEQUENTIAL`. Only the CPU memory arena was untested; disabling
it changed RSS by ~1 MB (noise). The RSS is driven by model size, not the arena.

## E10 — ONNX tree quantisation (reverted)

The 32 MB model is the RSS driver (E8), so the obvious lever is storing the trees in a
narrower type. It does not work with stock ONNX Runtime: the tree kernels are templated
on `float`, and the attribute readers call `GetAnyVectorAttrsOrDefault<float>(...)`,
which **rejects** `float16`/`int8` `*_as_tensor` attributes for `nodes_values`,
`target_weights`, etc. on both ORT 1.30 and 1.29.

One trim *was* possible: the all-zero `nodes_hitrates` attribute is unused and could be
dropped, taking the file **32.44 → 28.35 MB (−12.6%, bit-identical output)** and warm RSS
**396 → 389 MB**. Because the same binary block is re-serialised far more effectively by
E11, this change was **reverted** rather than shipped. The only E10 artefact kept is an
unrelated dependency fix (`scikit-learn` is required by `lgb.LGBMRegressor`).

## E11 — compile the trees to C#, drop ONNX Runtime (adopted)

The RSS is dominated by the ONNX Runtime native library plus its flattened node table,
not by the 32 MB model file alone (E8). The only way to move it is to stop using ORT.

`python/export_binary.py` re-serialises the two `TreeEnsembleRegressor` nodes out of
`model.onnx` into a flat `server/models/model.bin` — **no retraining**, a pure
re-serialisation of the already-verified model:

```
magic char[4] "OSRT" | version u32 | n_features u32 | n_targets u32
per target (graph order distance_m, duration_s):
  base_value f32 | n_trees u32 | n_nodes u32 | tree_offsets i32[n_trees+1]
  feature i32[n] | threshold f32[n] | left i32[n] | right i32[n] | value f32[n]
  is_leaf u8[n] | default_left u8[n]
```

The server replaces `InferenceSession` with `TreeEnsembleModel` (`server/TreeEnsembleModel.cs`),
a ~150-line, allocation-free walker: for every tree, follow `x <= threshold` to the left
child until a leaf, summing the leaf value onto the base. This is exactly ONNX
`BRANCH_LEQ` semantics. Node child ids are rebased to **global** rows at export time (the
ONNX attributes store *per-tree* node ids — the first implementation forgot this and
silently read the wrong leaf for every tree after the first).

**Correctness.** The extracted arrays are byte-identical to the ONNX attributes
(feature/threshold/left/right/leaf/value all `array_equal`). End-to-end the interpreter
reproduces ONNX Runtime to within float32 summation-order noise — **max |Δ| 0.07 m** on
distance and **0.005 s** on duration over 20k random pairs (≈3×10⁻⁶ relative). It is
*not* bit-exact (ORT sums the 400 leaves in a different order), but the residual is two
orders of magnitude below the service's 1-decimal output rounding and the 0.11 golden
tolerance. The 38-test suite passes unchanged on `model.bin`.

### Measured effect (both servers built and run on this box, back to back)

Single route, `curl`-warm, then 3,000 requests reading the `Server-Timing` header, plus
`wrk -t8 -c64 -d15s`; RSS read from `/proc/<pid>/smaps_rollup` after load:

| metric | ONNX Runtime | compiled trees (C#) | change |
|---|---|---|---|
| **RSS** (VmRSS) | 428 MB | **136 MB** | **−68% (3.1×)** |
| **Pss** (shared-page adjusted) | 391 MB | **99 MB** | **−75% (3.9×)** |
| anonymous heap (`Private_Dirty`) | 359 MB | **85 MB** | **−76% (4.2×)** |
| peak high-water (`VmHWM`) | 437 MB | **146 MB** | **−67%** |
| server-side latency p50 | 118 µs | **94 µs** | −20% |
| server-side latency p99 | 436 µs | **361 µs** | −17% |
| server-side latency p99.9 | 1,306 µs | **1,109 µs** | −15% |
| `wrk` throughput (single route) | 44,498 req/s | 40,584 req/s | within client-bound noise |
| on-disk artifact | 32.44 MB (`model.onnx`) | **17.97 MB** (`model.bin`) | −45% |

**Adopted.** This is the first change that materially moves RSS (the plan's headline
constraint). Server-side latency improves at every percentile; `wrk` throughput is
dominated by the HTTP client on this box and is comparable run-to-run. The 38-test
suite still passes.

### Honest costs / limits

- **RSS is still far above the plan's `< 30 MB` target — and the model is a larger share
  of it than this section first claimed.** Re-measured at the lighter load profile used by
  the capacity-price subsection (3,000 concurrent warm requests, no `wrk` run) the shipped
  511-leaf server sits at **123 MB**, vs the 136 MB in the table above, which was read after
  a 15 s `wrk` run; both are the same binary and the ~90 MB floor is load-independent. The
  host floor is ~90 MB, and on
  top of that the tree table costs **~1.86 MB of RSS per MB of `model.bin`**, not the ~1×
  implied by "the trees are only 18 MB" (measured at four capacities; see the capacity-price
  subsection above). The extra factor is `TreeEnsembleModel.Load` calling
  `File.ReadAllBytes`: the 18 MB file lands on the Large Object Heap as a *second*, full-size
  copy of a table that is then parsed into ~18 MB of typed node arrays, and that buffer stays
  resident until a gen2 collection. So capacity is not weakly coupled to RSS, it is a strong
  linear term. Getting to single-digit MB still requires a non-.NET host — out of scope —
  but streaming the load straight into the final arrays is an untaken ~15–18 MB that costs
  no accuracy (not measured here; the loader change was scoped out).
- **`model.bin` is a new tracked 18 MB artifact.** It is derived deterministically from
  `model.onnx` (`uv run python/export_binary.py`), which remains tracked as the source of
  truth.
- **`wrk` throughput did not improve** (40.6k vs 44.5k req/s); the win is RSS and
  per-request latency, not client-bound RPS.


## Experiment 0 — oracle snap: is snapping the short-trip bottleneck?

The model is asked to do two jobs at once: snap raw coordinates onto the network, and
route between them. E5 showed off-network data helps a lot, which raises the question
of whether the *residual* off-network error is snap uncertainty (fixable with a
snapper) or something else.

Experiment 0 gives the model the **exact** snap OSRM itself uses — `snap_endpoints.py`
calls `/nearest` once per unique endpoint at build time, so at inference the model
knows precisely where OSRM would have put each point. That is the ceiling any snapper
could reach. `experiment_oracle_snap.py` then trains two feature sets on the identical
split (fixed balanced test set, seeds 42/43/44 vary training only):

- `raw` — the shipped 8 base features;
- `oracle_snap` — base + exact snap displacement (east/north) at both ends + the
  snapped-point separation and bearing (16 features).

Primary test set is the off-network-short set (raw coordinates — the API's real input,
N≈2.8k); a balanced grid set is the control.

| test | features | dist MedAPE | dur MedAPE |
|---|---|---|---|
| off-short | `raw` | 38.9% | 38.3% |
| off-short | `oracle_snap` | **35.8%** | **34.1%** |
| grid | `raw` | 5.19% | 3.68% |
| grid | `oracle_snap` | 5.18% | 3.50% |

**Verdict: snapping is real but small — it is not the gap.** The exact snap cuts
off-network short-trip error by ~10% relative (≈3–4 MedAPE points), and leaves
~34–36% — about **7× worse** than the on-network grid set. The grid set is unchanged
(those points are already on the network, so snap ≈ 0). One honest wrinkle: on the
off-short `< 1 km` *distance* bucket the exact snap is slightly *worse* (77% → 93%
MedAPE) — revealing the true snapped separation sharpens the denominator of a
sub-kilometre relative error. Conclusion: **do not build a snapper** — the ceiling it
could reach is ~4 points, not the 30-point gap.

## Experiment 1 — connectivity / detour features

If snap is not the residual, the remaining candidate is detour: two raw points that are
close in a straight line can be far apart by road when a river, canal, railway or
motorway forces a detour. Those barriers are static geometry, so they can be revealed
with O(1) raster lookups.

`build_connectivity_raster.py` parses the OSM extract once (pyosmium) into a ~111 m
raster: nearest-road class per cell, water / rail / motorway barrier masks, and a
distance transform to the nearest barrier. `experiment_connectivity.py` adds, per pair:
endpoint road classes, endpoint barrier distances, barrier *crossings* along the
straight segment (water / rail / motorway runs), and road coverage on the segment
(fraction on-road, longest off-road gap in metres, #runs). Four feature sets, same
split and seeds (a second, independent run from Experiment 0's file — LightGBM
multi-thread training makes the two differ by < 1 point on every cell):

| test | features | dist MedAPE | dur MedAPE | grid dist | grid dur |
|---|---|---|---|---|---|
| off-short | `raw` | 39.1% | 38.6% | 5.2% | 3.7% |
| off-short | `conn` | 38.3% | 36.0% | 5.4% | 3.9% |
| off-short | `oracle_snap` | 35.5% | 34.6% | 5.2% | 3.5% |
| off-short | `oracle_snap_conn` | **34.0%** | **33.2%** | 5.2% | 3.6% |

The model-independent diagnostic agrees: within the off-short set, low straight-line
road coverage predicts high detour (the `road frac` → `med detour` table printed by the
script), so the feature is carrying real signal — just not much of it.

**Verdict: connectivity adds ~1–2 points on top of snap, and slightly *hurts* the grid
set** (distance 5.2% → 5.4% for `conn`). Stacked on the exact snap it reaches ~33–34%
— still ~8× the on-network level. Confirmed negative for shipping: no feature-contract
change, model unchanged.

### Why the gap does not close

Neither the exact snap (Exp 0) nor static barrier geometry (Exp 1) explains the
raw-coordinate short-trip error. Together they move it 39% → 34% (≈13% relative) and
stop there. The residual is genuine route choice and network topology — one-way
systems, turn restrictions, river crossings with few bridges — which is exactly the
discontinuous structure a smooth coordinate-plus-static-raster model cannot encode.
This falsifies the earlier README claim that the residual is "largely the snapping
floor": snapping accounts for ~4 of ~34 points.

### Negative sub-result: up-weighting off-network rows

`--off-boost` multiplies the sample weight of off-network rows (they are ~0.1% of the
pool). Since E1 already up-weights short trips, this double-weights them and only ever
hurts the off-short test, monotonically:

| off-boost | off-short dist | off-short dur | grid dist |
|---|---|---|---|
| 1 (none) | 38.8% | 37.8% | 5.17% |
| 25 | 40.7% | 39.5% | 5.22% |
| 100 | 44.7% | 41.3% | 5.29% |
| 400 | 48.4% | 43.1% | 5.69% |

**Rejected.** The default stays at 1.0.

### Cost

The full 4-set × 3-seed connectivity sweep is ~1,135 s; the 2-set × 3-seed oracle run
is ~378 s. The original harness rebuilt the feature matrix inside the seed loop, so
even though the split is fixed and the matrix is seed-independent it was materialised
3× per (kind, dataset). Both scripts now build `X` **once per kind** and reuse it across
seeds (and `--quick` gives a one-seed `raw`+`conn` run to check whether a feature moves
at all before paying for the full sweep). The one-off raster build
(`build_connectivity_raster.py`, osmium parse of the 51 MB PBF + distance transform) is
minutes on first run. Neither was needed to reach the verdict above.

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

### E7 — the capacity price: accuracy bought, memory paid (measured 2026-09-20)

E7 above adopted 511 leaves on the evidence that capacity keeps helping, but the
results log's baseline row also differs in *data* (it predates E1/E5), so it never
isolated what the extra leaves cost or bought. This run does: **`num_leaves` is the only
variable.** Protocol is the same as `experiments/try_s1_fair.py` — train on
`samples.parquet` + 80% of `offnetwork.parquet` with E1 inverse-frequency weights, 400
trees, fixed seed, and score on the unseen 20% (31,761 raw-coordinate pairs, i.e. what
the API is actually handed). Script: `experiments/scratch/price_capacity.py`.

| `num_leaves` | nodes/target | distance MedAPE | distance MAE | duration MedAPE | duration MAE | fit s |
|---|---|---|---|---|---|---|
| 63 | 50,000 | 4.51% | 1,822 m | 4.41% | 114 s | 36 |
| 127 | 101,200 | 3.84% | 1,564 m | 3.63% | 95 s | 46 |
| 255 | 203,600 | 3.15% | 1,322 m | 2.98% | 80 s | 67 |
| **511** | 408,400 | **2.62%** | **1,145 m** | **2.42%** | **66 s** | 78 |

The 511 row reproduces the independently published `try_s1_fair.py` figure (2.6% /
2.4%, MAE 1,140 m / 66 s), which is a useful cross-check that both harnesses score the
same thing. Note the curve is **saturating**: 63 → 255 leaves buys 1.36 MedAPE points
for +6.8 MB of artifact, while 255 → 511 buys only 0.53 points for +9.0 MB.

Memory for the same four capacities, measured the E11 way (3,000 concurrent warm
requests, then `/proc/<pid>/smaps_rollup`, median of 3 runs; `MODEL_PATH` swapped on a
single build so the runtime is byte-identical). Rows marked † are size-matched synthetic
tables (`experiments/scratch/make_sized_model.py`) — same node count, so the loader
allocates identically — verified against the two real artifacts at the endpoints
(63: real 93.5 MB RSS / 71.7 Pss vs synthetic 93.6 / 71.7; 511: real 123.3 / 101.3 vs
synthetic 123.0 / 101.1).

| `num_leaves` | `model.bin` | Pss | RSS | VmHWM | model cost over floor |
|---|---|---|---|---|---|
| 63 | 2.20 MB | 71.7 MB | 93.6 MB | 91.2 MB | 4.1 MB |
| 127 † | 4.46 MB | 75.4 MB | 97.3 MB | 95.1 MB | 7.8 MB |
| 255 † | 8.96 MB | 83.9 MB | 105.8 MB | 103.4 MB | 16.3 MB |
| **511** | 17.97 MB | 101.1 MB | 123.0 MB | 120.7 MB | 33.5 MB |

Fitted across all four points, **RSS ≈ 89.5 MB + 1.86 × `model.bin` MB** (residual
< 0.5 MB; the per-step ratios are 1.6–1.9×). That is the number E11 did not have: see
the correction in its costs below.

**Verdict — the 511-leaf capacity is justified, but it is the most expensive accuracy in
the model.** 8.17× more artifact (2.20 → 17.97 MB) buys 1.72× better distance MedAPE and
1.82× better duration, and removes 677 m / 48 s of MAE per pair; the price is **+29.4 MB
RSS** (93.6 → 123.0 MB). The saturating shape means 255 leaves is the interesting
fallback if memory ever binds more tightly than accuracy (+16.3 MB instead of +33.5 MB for
3.15% / 2.98%). Note that dropping to 63 leaves still leaves the box at 93.6 MB — **3×
over the plan's `< 30 MB` target** — so capacity is a poor lever for the footprint goal
and a good lever for accuracy.

Two caveats. First, this is the *capacity-only* price; the two artifacts actually
shipped also differ in data (E1/E5), and measuring the as-shipped pair on fresh
OSRM-labelled pairs puts the real-world gap wider (13.5% → 9.5% distance MedAPE, and
`< 1 km` 227% → 66%; `experiments/scratch/version_bakeoff.py`). Second, the shipped
63-leaf artifact predates E5, so off-network coordinates are out-of-distribution for it —
faithful to what it would do in production, but not a clean ablation, which is why the
controlled rows above are the ones to quote.

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
uv run snap_endpoints.py --url http://localhost:5001                             # Exp 0 data (needs live OSRM)
uv run experiment_oracle_snap.py --seeds 42 43 44                                # Exp 0
uv run build_connectivity_raster.py                                              # Exp 1 raster (needs the PBF)
uv run experiment_connectivity.py --seeds 42 43 44                               # Exp 1
uv run train_export_onnx.py --threads 16                                         # ship
```
