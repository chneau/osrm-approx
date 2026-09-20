# Simplification review — is the OSRM approximation service too complicated?

A design review of the repository, asking whether the current architecture is more
complex than the problem needs, and laying out **six completely different alternative
architectures** with their trade-offs.

This document is analysis, not a decision. Nothing here is implemented; the shipped
service is unchanged.

---

## 1. What the code actually is

The **algorithm at the centre of the service is tiny**. `server/TreeEnsembleModel.Predict`
is roughly 30 lines: walk each tree from `x <= threshold`, sum leaf values onto a base,
return two floats. The feature vector is 8 numbers (`Python`/`C#` contract).

Everything else is scaffolding around it:

| Layer | Size / artefacts | Notes |
|---|---|---|
| Runtime inference | `server/TreeEnsembleModel.cs` (~206 lines) + `server/Program.cs` (~160) | `model.bin` custom binary |
| Training + ONNX export | `python/train_export_onnx.py` (~465 lines) | LightGBM, ONNX, base calibration, float32 flooring |
| ONNX → binary compiler | `python/export_binary.py` (~174 lines) | `model.onnx` (31 MB) → `model.bin` (18 MB) |
| Ground-truth pipeline | `generate_grid.py`, `fetch_osrm_matrix.py`, `gen_offnetwork.py` (~550 lines) | Docker + OSRM `/table` |
| Research harness | `experiments.py`, `experiment_e5.py`, `experiment_oracle_snap.py`, `experiment_connectivity.py`, raster builders (~1,500 lines) | E1–E11, Exp 0/1 |
| Docs | `README.md`, `IMPROVEMENTS.md` (26 KB), this file | Very thorough |

So the complexity is **not** in the serving algorithm. It is in:
1. the offline data pipeline (justified — it produces the ground truth);
2. the **ONNX → custom-binary bridge** (largely accidental — workarounds for
   `ai.onnx.ml.TreeEnsembleRegressor` quirks);
3. the **research harness** (valuable, but not part of the product).

### Complexity audit — essential vs. accidental

| Item | Complexity | Does it earn its keep? |
|---|---|---|
| `TreeEnsembleModel.cs` + `model.bin` | ~200 + 174 lines, bespoke endianness parser | **Only because** ONNX Runtime cost 428 MB RSS (E11). Avoidable by other means (S4). |
| `calibrate_base_value()` | measures LightGBM's boost-from-average offset empirically | **Accidental.** Exists only because `TreeEnsembleRegressor` has no initial score. |
| `floor_float32()` | rounds thresholds down to float32 | **Accidental.** Exists only because ONNX stores float32 thresholds while LightGBM compares float64. |
| `model.onnx` + `model.bin` + `model_metadata.json` | 3 tracked artifacts, one derived | **Accidental** two-artefact coupling (source of truth vs. served). |
| Chunked `/table` caller | ~339 lines, retry/bisect logic | Essential for generating labels, but the real work is OSRM's. |
| Experiment harness + rasters | ~1,500 lines, 32 MB of intermediate data | Genuinely good research; **not** part of the product. |
| The model itself | 511 leaves × 400 trees × 2 targets | Buys ~5% MedAPE over a haversine baseline. |

### Headline observation

> The tree ensemble is a **lossy, trained compression of an all-pairs distance matrix
> that was already computed offline and then thrown away.**

`samples.parquet` (30 MB) is the all-pairs set of 2,203 routable grid points
(4,850,998 pairs). The two shipped regressors are, in effect, a smooth interpolator fit
to that matrix. The product and the pipeline can be separated much more aggressively than
they currently are.

### Accuracy context (from README / IMPROVEMENTS)

- Served model vs. live OSRM: MedAPE **4.9% (duration) / 4.1% (distance)** on random pairs.
- Band-balanced: **9.9% / 10.3%**.
- `< 1 km` band: **78% / 74%** — inherent to any smooth coordinate function; Experiments 0
  and 1 show snapping and static barrier rasters only move it ~39% → ~34% and stop.
- Haversine-as-distance baseline: 29.1% MedAPE.

The residual short-trip error is genuine network topology, not a modelling gap. That
matters for choosing an alternative: only an exact/oracle method (S2) can attack it.

---

## 2. Alternative architectures

Ordered from simplest to most different. All reuse the **same offline ground truth**
(`samples.parquet`) and can be scored with the existing `python/experiments.py` /
`tests/osrm_vs_onnx.py` harness for a like-for-like comparison.

### S0 — Do we need it at all? Memoizing proxy in front of OSRM

Not "outside the box", but the philosophical baseline. If query traffic is repetitive,
put a cache in front of `osrm-routed`: exact answers, zero modelling, zero drift, ~50
lines. Cost: OSRM's 574 MB and ~4 ms p50. Worth stating because it reframes the project
as a **memory/latency vs. exactness** trade, which is the only thing the ML stack buys.

### S1 — Ship the matrix, interpolate (drop the model, keep the data)

The all-pairs matrix *is* the model. Lay the nodes on a **regular lat/lon lattice** and
store `distance_m` and `duration_s` as two `(N,N)` arrays (~39 MB at float32, ~20 MB at
float16/bucketed). At query time compute fractional lattice coordinates for both
endpoints and take the **4-D bilinear blend of a 4×4 block**:

```
D(A,B) ≈ Σ_i Σ_j  wa_i * wb_j * D[ia_i, jb_j]      # 16 lookups, ~40 FLOPs
```

Properties:

- **Zero training, zero ONNX, zero binary format, zero calibration.** Serving code is
  smaller than `RoutingFeatures.Compute`.
- On-lattice values are **exact OSRM**, not a 5%-error fit.
- Off-lattice error is bounded by cell size; bilinear is a natural smoother, and can be
  made multi-resolution (fine cells in the core) just like the current ring grid.
- Memory scales as **`N²`** (finer grid = quadratically more), whereas the tree model is
  roughly `O(cells)` and interpolates smoothly by construction. This is the one reason a
  learned model may still be preferred.

**Best when:** minimum code and minimum moving parts matter most, and ~1 km lattice
resolution is acceptable.

### S2 — Precomputed distance oracle (hub labels / contraction hierarchies)

Invert the trade: **be exact, not smooth.**

- Build a sparse graph over grid nodes (or, better, over the real OSM road graph) with
  OSRM edge weights.
- Compute **Pruned Landmark Labeling / hub labels** once. Each node gets a handful of
  `(landmark, distance)` labels; a query is `min` over the intersection of labels.
- Query is exact, ~microseconds, **no per-query graph traversal** (satisfies the original
  plan constraint), and does **not smooth over short-trip discontinuities** — exactly
  where the current model fails (`< 1 km` MedAPE 74–78%).

This attacks the real limitation rather than the serving stack. Different *kind* of
complexity (graph algorithms + a one-off build), but a simpler and **more accurate**
runtime. Cost: materialise the road graph once (the `osrm-extract` step already does).
Mature hub-labeling implementations exist in C++/Rust.

**Best when:** short-trip exactness and low memory are the binding constraints.

### S3 — Closed-form detour + speed model (drop the ML entirely)

Fit a handful of coefficients by least squares on the same samples:

```
hav   = haversine(A,B)
theta = bearing(A,B)
detour(hav, theta, pos) = a0 + a1*hav + a2*sin(theta) + a3*cos(theta) + a4*sin(2θ) + ...
distance ≈ hav * max(1, detour)
duration ≈ distance / v(pos, theta)          # v = a few Fourier/polynomial terms
```

- Model artifact = a dozen floats, optionally inlined as constants.
- Online = one trig eval + ~20 FLOPs. No trees, no files, no parser, trivially auditable,
  impossible to blow up RSS.
- Accuracy: expect a few points worse on long trips and materially worse on short ones —
  but quantify it with the existing harness before committing. Given the E1–E11 / Exp 0/1
  findings, a smooth closed form may be close to the practical ceiling of any
  coordinate-only model.

**Best when:** the accuracy target is "a few percent on the trips that matter" and a
boring, auditable implementation is preferred.

### S4 — Keep the GBM, delete the ONNX bridge and the custom format

ONNX is not needed as an intermediate. The complexity in `train_export_onnx.py`
(`calibrate_base_value`, `floor_float32`, IR-version fiddling) and `export_binary.py`
exists only because of `TreeEnsembleRegressor`'s storage choices.

- Train → dump the booster to **LightGBM's own text model** (`booster.dump_model()` /
  `save_model`), or directly to a flat array format. Write one small C# parser.
- Delete `model.onnx`, delete `export_binary.py`, delete the float32-threshold and
  base-calibration logic (keep thresholds as float64 in your own format if desired).
- Optionally **codegen** traversal into a `.g.cs` at build time so there is no runtime
  parser at all.

Keeps today's accuracy, removes the most fragile part of the pipeline. If RSS was the
only reason ONNX was dropped (E11), regenerating `model.bin` straight from the LightGBM
dump removes a whole stage and a correctness class.

**Best when:** you want current accuracy but lower pipeline risk.

### S5 — Hybrid: analytic baseline + tiny residual learner

Combine S3 and S4. Compute the deterministic baseline (`hav × detour`, `distance / speed`)
and train a **small** tree ensemble or k-NN only on `log(residual)`. Because most of the
structure is already captured, far fewer leaves/trees are needed (perhaps the 63-leaf
3.8 MB model) to reach the same accuracy — smaller model, faster walk, and the heavy
lifting is inspectable arithmetic.

**Best when:** you want the best accuracy-per-byte with simple serving.

### S6 — k-NN / kernel regression over the training pairs

Store the labelled pairs; at query average the `k` nearest in the 4-D `(orig, dest)`
space (KD-tree/ball-tree). Conceptually the simplest "learn from data" option, and **S1
is its `k=1`, grid-aligned, O(1) special case**. In practice likely more memory and slower
than S1 with similar accuracy — reach for it only if the grid is genuinely irregular.

**Best when:** the sampling grid is irregular and you cannot resample onto a lattice.

---

## 3. Comparison

| Option | Model artefact | Online cost | Exactness | Short-trip accuracy | Main risk |
|---|---|---|---|---|---|
| S0 cache OSRM | none (OSRM graph) | ~4 ms p50 | exact | exact | 574 MB RSS, slow p50 |
| S1 matrix + bilinear | `(N,N)` arrays (~40 MB) | 16 lookups | exact on lattice | cell-size bound | `O(N²)` memory |
| S2 hub labels | labels (~small) | label intersection | exact on graph | exact | build complexity |
| S3 closed form | ~12 floats | ~20 FLOPs | smooth approx | poor | accuracy target |
| S4 direct GBM dump | flat arrays (~18 MB) | tree walk | exact to GBM | same as today | keeps ML pipeline |
| S5 baseline + residual | small arrays | baseline + small walk | smooth approx | same as today | mid complexity |
| S6 k-NN | training pairs | KD-tree query | local average | grid-dependent | memory/latency |

## 4. Recommendation

| If your binding constraint is… | Pick |
|---|---|
| Minimum code / minimum moving parts | **S3** (closed form) or **S1** (matrix + bilinear) |
| Memory and exactness on short trips | **S2** (hub labels) |
| Keep current accuracy, shed pipeline risk | **S4** (dump LightGBM directly, delete ONNX) |
| Best accuracy per byte with simple serving | **S5** (baseline + residual) |
| "Do we need this at all?" | **S0** (cache OSRM) |

**Suggested first step:** prototype **S1 and S3** against the existing
`python/experiments.py` / `tests/osrm_vs_onnx.py` harness. They reuse the exact
ground truth and metrics already trusted, so you get a like-for-like MedAPE (overall and
per band) for a fraction of the code. If S1 reproduces the tree model's numbers, the
entire `train_export_onnx.py` + `export_binary.py` + `TreeEnsembleModel.cs` + `model.*`
stack disappears. If it does not, the gap tells you exactly what the trees are buying.

One caveat that applies to every option except S2: the current design's hardest problem —
short **raw-coordinate** trips — is not solved by any of them. Per Experiments 0/1 it is
probably unsolvable by any coordinate-only method, because it is genuine route-choice and
network topology. S2 is the only proposal here that stops approximating a discontinuous
function with a smooth one.

---

## Appendix A — measured S1 prototype + fixed-point encoding

Prototypes: `experiments/try_s1.py`, `experiments/try_s1_fair.py`. The first ONNX
comparison was **contaminated** — the shipped model was trained on
`data/processed/offnetwork.parquet`, so scoring it there flatters it. `try_s1_fair.py`
retrains a 511-leaf / 400-tree GBM on `samples.parquet` + 80% of the off-network pairs,
holds out the other 20%, and scores both models on the **same unseen raw coordinates**.

### Head-to-head (raw / off-network coordinates, N=31,761)

| model | target | Median APE | MAE | 1–3 km MedAPE |
|---|---|---|---|---|
| **GBM** (honest, never saw the test rows) | distance | **2.6%** | **1,140 m** | 32% |
| **S1** nearest grid-node lookup (2,203 nodes) | distance | 3.4% | 1,945 m | 100% |
| **GBM** (honest) | duration | **2.4%** | **66 s** | 2.4% |
| **S1** nearest grid-node lookup | duration | 5.0% | 166 s | 4.7% |

S1 with `k=4/8` inverse-distance-weighted interpolation did **not** improve on nearest-node
lookup. S1 is viable and far simpler, but the GBM is genuinely better: ~1.7× lower
distance MAE and ~2.5× lower duration MAE. The gap is the off-network snapping the GBM
learned (E5) and a fixed-node matrix cannot represent; the `1–3 km` distance column is the
clearest symptom.

### Fixed-point coordinate encoding (Greater Manchester bbox, offset from origin)

| scale | precision | lat span | lon span | bits needed | signed int24? |
|---|---|---|---|---|---|
| 1e-4 | ~11 m | 3,500 | 8,699 | 14 | yes |
| 1e-5 | ~1.1 m | 35,000 | 86,999 | 17 | yes |
| 1e-6 | ~11 cm | 350,000 | 869,999 | 20 | yes |
| 1e-7 | ~1.1 cm | 3,500,000 | 8,699,999 | 24 | no (needs unsigned 24) |

So 24-bit offsets are fine for a bounded region and overkill at ~1 m (17 bits). Global
lat/lon at 1e-5 needs 25 bits signed and does **not** fit int24.

### Where the memory actually is

For S1 the coordinates are `O(N)` and the matrix is `O(N²)`. At N=2,203, 24-bit
coordinates cost ~13 KB versus a 39 MB matrix — packing them saves nothing. Quantise the
**matrix** instead:

| matrix encoding (2 targets, N=2,203) | size | accuracy cost |
|---|---|---|
| float32 | 38.8 MB | — |
| **uint16, distance in decametres + duration in seconds** | **19.4 MB** | max 5 m / mean 2.5 m, zero overflow |

(`uint16` metres overflowed on 0.0044% of pairs — max distance 70,124 m — so decametres
are required.)

### Scaling wall

At the current ~1 km lattice `N=2,203`. A 250 m lattice is ~35,000 nodes, so the matrix is
~1.2 billion entries → ~2.5 GB even at `uint16`. Fixed-point compression buys 2–4×, not
the ~16× needed. This is the structural reason a learned model can beat a stored matrix.

### Verdict

Adopt S1 only if the short-trip regression is acceptable; then use `uint16`
decametre+second matrix values (the real saving) and optional 20/24-bit coordinate offsets.
If accuracy must hold, prefer **S4** (drop the ONNX bridge) or **S2** (exact oracle).
