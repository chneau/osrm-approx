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

## 3b. Measured results (consolidated)

The proposals above were prototyped and measured. All accuracy numbers are on the **same
held-out raw-coordinate split** (`offnetwork.parquet`, 20% hold-out, N=31,761) unless
noted; details and methodology are in Appendices A–C.

| method | distance MedAPE | duration MedAPE | artefact | verdict |
|---|---|---|---|---|
| GBM (shipped, retrained honestly) | **2.6%** | **2.4%** | 18 MB `model.bin` | baseline |
| **S1** matrix + nearest/interpolated lookup | **3.4%** | **5.0%** | ~39 MB (`uint16` → 19 MB) | **viable, simpler** |
| S2b full OSM driving graph (independent router) | 6.2–11.0% | 24–31% | 438k-node graph | rejected |
| S2 landmark oracle (on-grid, L=512) | 5.1% | — | 9 MB | rejected |
| S2 hub labels / PLL on proximity graph | 35.2% | 163.9% | 30 MB labels | rejected |
| S2b major-roads-only OSM | 29.6% (`<1 km` 58.8%) | 52.4% | small graph | rejected |

### Key findings

1. **S1 works** and is the only simplification worth shipping. It needs no training, no
   ONNX, no custom binary — just the all-pairs matrix and a lattice lookup. It trades a
   little accuracy (2.6% → 3.4% distance) for a large drop in moving parts.
2. **Fixed-point encoding:** 24-bit coordinates are unnecessary in S1 (coordinates are
   `O(N)`, the matrix is `O(N²)`; 24-bit coords save ~13 KB of 39 MB). The real win is
   `uint16` matrix values with distance in **decametres** + duration in seconds →
   **38.8 MB → 19.4 MB** at ~2.5 m mean error. A 1 m-precision coordinate fits in **17
   bits** as a bbox offset (Appendix A).
3. **S2's oracle machinery is correct and fast** (PLL == scipy, 3.9 s build, ~15 µs
   queries, 326 labels/node) but a **proximity graph overestimates OSRM by +35%** because
   summing pairwise shortest paths is only an upper bound. A true S2 needs the actual road
   graph. A landmark oracle was better (~5%) but still behind S1/GBM (Appendix B).
4. **A real simplified OSM graph is a different router, not an OSRM approximation.**
   It ignores turn restrictions (median distance ratio 0.90×), uses a different speed
   profile (duration 0.70–0.76×) and leaves 4–10% of pairs unreachable. Dropping local
   roads — the literal "simplification" — raises `< 1 km` distance error from 13% to 59%
   (Appendix C).
5. **The learned ensemble wins because it distilled OSRM's outputs** (snapping, turn
   rules, speed profile), which an independent graph must rediscover, i.e. reimplement
   OSRM. This is the strongest argument for keeping the ML approach or, if code size is
   the concern, keeping it while removing the ONNX bridge (S4).

## 4. Recommendation

| If your binding constraint is… | Pick | Measured outcome |
|---|---|---|
| Minimum code / minimum moving parts | **S1** (matrix + lookup) | 3.4% / 5.0% — viable |
| Keep current accuracy, shed pipeline risk | **S4** (dump LightGBM directly, delete ONNX) | keeps 2.6% / 2.4% |
| Best accuracy per byte with simple serving | **S5** (baseline + residual) | untested, plausible |
| Exact oracle without traversal | **S2** (hub labels) | needs the real road graph; not competitive as built |
| "Do we need this at all?" | **S0** (cache OSRM) | exact, 574 MB RSS |

**Suggested first step:** ship S1 if a 2.6% → 3.4% distance regression is acceptable — it
removes `train_export_onnx.py`, `export_binary.py`, `TreeEnsembleModel.cs` and both model
artefacts (`experiments/try_s1_fair.py` is the measured prototype). Otherwise keep the
model and do **S4** to delete the ONNX bridge.

Caveat: the current design's hardest problem — short **raw-coordinate** trips — is not
solved by any option here, and the OSM/S2 experiments show why: it is genuine route-choice
and network topology, not a coordinate-modelling gap. Only an exact road-network router
(S0, or S2 *with a real graph*) reproduces it.

---

## 5. Related work

> **Citation caveat:** this list was assembled from memory, not a live search. Author/venue
> keys are reliable enough to look up, but verify exact titles on Google Scholar / arXiv /
> the linked project before quoting them.

### 5.1 Exact distance oracles (the S2 family)

- **Contraction Hierarchies (CH)** — Geisberger, Sanders, Schultes, Delling (2008).
  Shortcut-based preprocessing + bidirectional upward search. What OSRM's `--algorithm ch`
  is.
- **Highway Hierarchies** — Sanders & Schultes (2005–2006). Predecessor of CH.
- **Highway Dimension** — Abraham, Fiat, Kaplan, Lucier, *Highway Dimension, Shortest
  Paths, and Provably Efficient Algorithms* (SODA 2010). **The theory that explains why
  road networks admit tiny search spaces and labels — and, by contrast, why a sampled
  proximity graph does not (Appendix B).**
- **Customizable Route Planning (CRP) / Multi-Level Dijkstra (MLD)** — Delling, Goldberg,
  Pajor, Werneck (SEA 2011; Transportation Science 2017). OSRM's default algorithm.
- **Hub Labeling** — Abraham, Delling, Goldberg, Werneck, *A Hub-Based Labeling Algorithm
  for Shortest Paths in Road Networks* (SEA 2011). The exact oracle S2 approximates.
- **Pruned Landmark Labeling (PLL)** — Akiba, Iwata, Yoshida (SIGMOD 2013). The algorithm
  implemented in `experiments/pll.c`.
- **ALT (A\* + landmarks)** — Goldberg & Harrelson (2005). Landmark lower bounds; the
  landmark-oracle experiment (Appendix B) is the upper-bound cousin.
- **PHAST / many-to-many** — Delling, Goldberg, Werneck (2011). Batched shortest paths,
  relevant to `/table`-style matrix generation.
- **Approximate distance oracles (general graphs)** — Thorup & Zwick (JACM 2005);
  Patrascu & Roditty (FOCS 2010). The `(2k−1)`-stretch / `O(n^{1+1/k})`-space frontier
  that road networks beat because of bounded highway dimension.

### 5.2 Practical routing engines

- **OSRM** — `Project-OSRM/osrm-backend` (CH + MLD, `/table`).
- **Valhalla** — `valhalla/valhalla` (tiled hierarchy, dynamic costing, isochrones).
- **GraphHopper** — `graphhopper/graphhopper` (CH/ALT, Java).
- **RoutingKit** — Karlsruhe (CH, CRP, PHAST), C++ reference implementations.
- **r5** — `conveyal/r5` (fast many-to-many matrices, transport planning).
- **pgRouting** — PostGIS extension.

### 5.3 Approximate distance estimation from coordinates

Closest academic framing to this repo (S1/S3, and the tree ensemble).

- **Landmark / sketch distance estimation** — Potamias, Bonchi, Castillo, Gionis, *Fast
  Shortest Path Distance Estimation in Large Networks* (CIKM 2009).
- **Low-dimensional metric embeddings of road networks** — search "low distortion embedding
  road network distance" / "Euclidean embedding road networks" (Abraham, Bartal, Neiman
  and follow-ups). A linear/embedding alternative to the tree ensemble.
- **Spatial interpolation of travel-time / OD matrices** — kriging and Gaussian-process
  models for OD travel time; the academic version of S1's matrix + interpolation.
- **"Network distance from Euclidean distance" / detour-factor models** — recurring
  GIScience topic; the analytic baseline in S3.
- **Isochrone / travel-time rasters** — Valhalla & ORS isochrones; accessibility literature.

### 5.4 Learned / neural approaches

- **ETA / travel-time regression** — *Learning to Estimate the Travel Time* (Wang et al.,
  KDD 2018); Uber's **DeepETA** (engineering blog, 2022); Google Maps' GNN traffic
  prediction. Note: these predict on routes/trajectory features, not raw coordinate pairs.
- **GNNs for shortest paths / distance** — *Neural Bellman-Ford Networks* (ICLR 2022);
  *A\*Net* (ICLR 2023); *Path Planning using Neural A\* Search* (Yonetani et al., ICML
  2021).
- **"Neural distance oracle"** — search this exact phrase on arXiv/Scholar; there is
  2022–2024 work here, though I can't name a specific paper confidently.
- **Amortized shortest path / learned oracles** — search "learned distance oracle",
  "amortized shortest path".

**Honest gap:** I'm not aware of a well-known project that specifically mimics **OSRM
outputs from raw coordinates with a GBM**, which is what this repo does. The closest
framings are travel-time estimation and learned distance oracles.

### 5.5 Integer / fixed-point geospatial encodings

Relevant to the int24 question (Appendix A): Google's **S2 geometry** (64-bit fixed point
on the sphere) and Uber's **H3** (hexagonal integer indexing). Both use integer
quantisation, but much coarser than 24-bit and not distance oracles.

### 5.6 Benchmarks and data

- **DIMACS Implementation Challenge — 9th: Shortest Paths** (2006): canonical road-network
  datasets and query sets; the standard for comparing CH/ALT/hub labeling.
- **PACE** challenge (check the 2024 edition's scope).
- **Transportation Networks for Research (TNTP)**: classic networks with turn penalties.
- **OpenStreetMap + OSRM** as open ground truth (used here).
- **RoutingKit / Karlsruhe benchmark pages**.

### 5.7 How the measured findings map to the literature

| Finding here | Literature equivalent |
|---|---|
| Tree ensemble (8 features → dist/dur) | Amortized learned distance oracle; ETA/travel-time regression |
| S1 all-pairs matrix + interpolation | OD-matrix interpolation / kriging; truncated exact oracle |
| S2 hub labels / PLL | Hub labeling (Abraham 2011), PLL (Akiba 2013) |
| S2 proximity graph overestimates +35% | Highway-dimension theory (Abraham 2010); segments, not sampled metrics |
| S2 landmark oracle ≈5% | ALT / landmark distance estimation (Goldberg-Harrelson 2005; Potamias 2009) |
| Simplified OSM loses turn rules/speeds | Why engines keep edge-based graphs + turn tables (CH / CRP / MLD) |
| Distributed OSRM `/table` generation | PHAST / many-to-many (Delling 2011) |
| int24 fixed-point coordinates | S2 geometry / H3 integer encoding |

**Suggested reading order for this project:** (1) Abraham et al., Highway Dimension —
explains the S2 result; (2) Akiba et al., PLL — the algorithm implemented; (3) Delling
et al., CRP/MLD — what OSRM does; (4) Potamias et al. (2009) — canonical graph distance
estimation; (5) an ETA/prediction survey — the ML framing closest to the tree ensemble.

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

---

## Appendix B — measured S2 prototype (hub labels / PLL)

Prototypes: `experiments/build_s2_graph.py`, `experiments/pll.c`, `experiments/eval_s2.py`.

### What was built

1. A **multi-resolution node set** (5 km at ~222 m, 5–15 km at ~445 m, 15–40 km at ~890 m)
   snapped to a live OSRM; 11,699 routable nodes.
2. A **sparse proximity graph**: each node joined to its `k` nearest nodes, each edge
   weighted with the *exact* OSRM `/table` distance and duration (so edge weights are not
   the source of error). 38,905 undirected edges, average degree 6.7.
3. A **Pruned Landmark Labeling (PLL)** hub-label oracle in C (`experiments/pll.c`):
   labels built in 3.9 s, avg **326 labels/node**, max 603, **30.5 MB** raw for distance;
   a query is a two-pointer intersection, **~15 µs** in Python (sub-µs in C), no traversal.
4. Correctness: PLL matches `scipy` shortest paths on the same graph to **max 0.012 m**.

So the oracle machinery works. The problem is the **graph**.

### Raw-coordinate accuracy (same 20% `offnetwork.parquet` split)

| model | distance MedAPE | distance MAE | duration MedAPE | duration MAE |
|---|---|---|---|---|
| GBM (honest) | **2.6%** | **1,140 m** | **2.4%** | **66 s** |
| S1 grid lookup | 3.4% | 1,945 m | 5.0% | 166 s |
| **S2 / PLL** | 35.2% | 12,008 m | 163.9% | 3,390 s |

### Why S2 failed here: shortest paths are not concatenations

A graph whose edges are *shortest-path distances between sampled points* systematically
**overestimates** the true distance. For any path `u=x0..xm=v`, repeated triangle
inequality gives `Σ d(xi,xi+1) ≥ d(u,v)`; the excess is the grid discretisation error and
it accumulates along the route. Measured on the exact all-pairs matrix (2,203 nodes,
`experiments` k-sweep):

| k (neighbours/node) | edges | shortest-path / direct OSRM (median) | MedAPE |
|---|---|---|---|
| 4 | 8,804 | 1.58× | 58.1% |
| 8 | 17,616 | 1.22× | 22.8% |
| 16 | 35,240 | 1.06× | 11.5% |
| 32 | 70,488 | 0.99× | 7.8% |
| 64 | 140,984 | 0.94× | 7.0% |

The fine 11,699-node graph at k=6 shows the same effect (+35% median overestimate).
Denser graphs converge slowly and never become exact; a *complete* graph would be S1.

### The cheaper dodge: a landmark oracle

`min over L landmarks of d(u,l) + d(l,v)` using exact OSRM distances (no graph): on the
2,203-node grid, L=64 → MedAPE 6.2%, L=128 → 5.3%, L=512 → 5.1%, at `N×L` storage
(9 MB at L=512). Better than the proximity graph, still worse than the GBM/S1, and it
degrades to S1's matrix as L→N.

### Verdict — rejected in this form

Hub labels are fast, compact and *correct*, but their exactness is only as good as the
graph they run on. A graph of pairwise shortest-path weights is an upper bound that
diverges, so PLL on a sampled proximity graph cannot match the GBM. **A true S2 needs the
actual road graph** (road-segment edges, OSRM-consistent weights, one-ways and turn
restrictions) — i.e. reconstructing the very thing being approximated. That is a large
effort with a real chance of just re-deriving OSRM, so S2 is rejected unless the goal is
specifically an exact on-network oracle and the road graph can be imported directly
(e.g. OSM PBF + a routing library, accepting profile divergence from OSRM).

The reusable positive result: `experiments/pll.c` builds correct hub labels for an
11.7k-node graph in ~4 s and answers queries in microseconds, so the *oracle* half of S2
is de-risked if a real road graph ever becomes available.

---

## Appendix C — "can we just use a simplified OSM?"

Prototype: `experiments/try_osm_graph.py` (a PEP-723 `uv` script). It builds the **real
OSM driving network** with `pyrosm` (no OSRM, no training), weights each segment by
great-circle length and a `maxspeed`-derived duration, snaps raw query coordinates to the
nearest graph node, and runs shortest paths. This is an independent routing graph built
from the exact same OSM extract OSRM used.

- Full driving network: **438,191-node largest component**, 860,764 directed edges, built
  in **~4 s**.
- "Simplified" = keep only motorway/trunk/primary/secondary/tertiary (drop residential,
  unclassified, service).

### Results (same `offnetwork` / `offnetwork_short` raw-coordinate splits)

| graph | split | distance MedAPE | duration MedAPE | distance bias | routable |
|---|---|---|---|---|---|
| full OSM driving | offnetwork | 11.0% | 24.0% | 0.90× | 90.5% |
| full OSM driving | offnetwork_short | 6.2% | 30.6% | 0.98× | 95.8% |
| **major roads only** | offnetwork_short | **29.6%** | **52.4%** | 0.86× | 94.8% |
| major roads, `< 1 km` | offnetwork_short | **58.8%** | 53.5% | — | — |

Compare the learned model on the same task: **2.6% / 2.4%**; S1: 3.4% / 5.0%.

### Why it loses

1. **No turn restrictions.** The node-based OSM graph permits turns OSRM forbids, so it
   finds *shorter* paths than OSRM (median distance ratio **0.90×**) — not just noisy, but
   systematically optimistic.
2. **Different speed profile.** Segment speeds from `maxspeed`/road-class defaults give
   durations ~25–30% short (ratio **0.70–0.76×**); OSRM also adds turn costs and its own
   penalty model.
3. **Unreachable pairs.** 4–10% of raw-coordinate pairs return no route even after keeping
   the largest component (one-way/isolated-fragment/snap effects).
4. **Simplification is exactly what breaks short trips.** Dropping residential roads
   raises `< 1 km` distance error from 13% to **59%** and duration to 54% — the local
   network *is* the short-trip signal.

### Verdict — rejected

A simplified OSM graph is trivially buildable and pleasantly fast, but it is a **different
router**, not an OSRM approximation. To close the gap you must add turn restrictions,
match OSRM's access filtering, speed and turn-penalty profile, and snap to edges the way
OSRM does — that is reimplementing OSRM. The learned tree ensemble wins at this specific
task because it was trained on OSRM's *outputs*, so it distilled OSRM's snapping, turn
rules and speeds implicitly; a from-scratch graph has to rediscover them explicitly.

If the goal changes from "mimic OSRM" to "a decent independent router," the full OSM graph
is a reasonable starting point — but at that point just run OSRM, which is already that
router and is exact by definition.
