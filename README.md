# Fast OSRM Approximation Service — Greater Manchester

An ultra-fast `A → B` lookup that returns static `(duration_s, distance_m)` mimicking OSRM
car routing in Greater Manchester — without running OSRM at query time.

OSRM is used **offline only** to generate ground truth. The runtime is a small .NET 10
service that computes a handful of geometric features and evaluates two LightGBM tree
ensembles with a dependency-free C# interpreter (no ONNX Runtime on the request path).
There is no graph traversal, no routing engine and no external dependency on the
request path.

---

## How it works

```
OFFLINE (Python)                                  ONLINE (.NET 10)
────────────────                                  ────────────────
Geofabrik .osm.pbf                                GET /route?orig=lat,lon&dest=lat,lon
        │                                                  │
        ▼                                                  ▼
osrm-extract / partition / customize              haversine · bearing · deltas
        │                                                  │
        ▼                                                  ▼
multi-resolution grid (3000 pts)            compiled tree ensemble (1 pass, no ONNX RT)
   • 278 m core / 557 m mid / 1113 m fringe              ├── distance_m
        │                                                └── duration_s
        ▼                                                        │
OSRM /table (chunked) ──► samples.parquet                        ▼
   + random off-network coords ──► offnetwork.parquet  {"duration_s":..,"distance_m":..}
        │
        ▼
LightGBM ×2  (E1 weights, 511 leaves) ──► model.onnx ──► model.bin
```

Feature vector (order is a hard contract between `python/train_export_onnx.py` and
`server/Program.cs`):

| # | feature | notes |
|---|---------|-------|
| 0 | `orig_lat` | |
| 1 | `orig_lon` | |
| 2 | `dest_lat` | |
| 3 | `dest_lon` | |
| 4 | `haversine_dist_m` | great-circle distance |
| 5 | `bearing_deg` | initial bearing, `[0, 360)` |
| 6 | `lat_delta` | `dest_lat - orig_lat` |
| 7 | `lon_delta` | `dest_lon - orig_lon` |

Two LightGBM regressors (distance and duration) are flattened into `ai.onnx.ml`
`TreeEnsembleRegressor` nodes and share one graph. At export the nodes are re-serialised
into a compact `server/models/model.bin`, and the server evaluates the trees with a tiny
custom C# interpreter (`server/TreeEnsembleModel.cs`) — **no ONNX Runtime at serve time**
(that native dependency was the dominant RSS cost; see `IMPROVEMENTS.md` E11).
LightGBM's boost-from-average initial score is measured empirically and baked in as each
node's `base_values`. Split thresholds are rounded **down** to the largest `float32 ≤
threshold`, which makes the comparison exactly equivalent to LightGBM's `float64`
comparison (round-to-nearest could send an input down the wrong branch otherwise — see
`IMPROVEMENTS.md`).

The model is trained on all-pairs grid samples **plus** random off-network coordinates
(the service is handed raw coordinates, so it must learn how OSRM snaps them), with
inverse-frequency sample weights so short trips are not drowned out by long ones. See
`IMPROVEMENTS.md` for the full experiment log (E1–E11) behind that choice.

---

## Quickstart

Prerequisites: Docker, [uv](https://docs.astral.sh/uv/), .NET 10 SDK.

```bash
# 0. Download the Greater Manchester extract (~51 MB)
curl -L -o data/raw/greater-manchester.osm.pbf \
  https://download.geofabrik.de/europe/united-kingdom/england/greater-manchester-latest.osm.pbf

# 1. M0 — build the OSRM MLD graph (extract / partition / customize)
npm run osrm:init

# 2. M1 — start OSRM and generate the ground-truth sample set
npm run osrm:up
npm run data            # grid + chunked /table -> data/processed/samples.parquet
npm run data:offnetwork # random raw coords -> data/processed/offnetwork.parquet (E5)

# 3. M2 — train LightGBM and export server/models/model.onnx → model.bin
npm run train                     # writes model.onnx
uv run python/export_binary.py    # compiles model.onnx -> model.bin (E11)

# 4. M3 — serve
npm run serve           # listens on http://localhost:5080

# 5. M4 — benchmark latency / RSS / accuracy against live OSRM
npm run bench           # also writes tests/golden_routes.json
npm run test

# 5b. Full accuracy statistics vs live OSRM (standalone uv script)
npm run stats           # writes tests/OSRM_VS_ONNX.md + .json
npm run stats:stratified

# 6. Decommission OSRM
npm run osrm:down
```

### API

```bash
curl "http://localhost:5080/route?orig=53.4808,-2.2426&dest=53.3537,-2.2749"
# {"duration_s":1407.5,"distance_m":18643.7}
```

`GET /health` reports the loaded model and the feature contract. Every response carries a
`Server-Timing: app;dur=<ms>` header so server-side processing time can be measured without
the HTTP client's own overhead dominating the number.

---

## Repository layout

```
├── docker-compose.yml              # OSRM MLD build + serving (offline generation only)
├── package.json                    # runner scripts
├── IMPROVEMENTS.md                 # model-improvement log (E1–E9), good and bad results
├── SIMPLIFICATION.md               # design review + measured alternative architectures (S1/S2/OSM)
├── data/
│   ├── raw/                        # greater-manchester.osm.pbf, .osrm* artefacts
│   └── processed/                  # grid.parquet, samples.parquet, offnetwork.parquet
├── python/                         # self-contained uv scripts (PEP 723 inline metadata)
│   ├── generate_grid.py            # multi-resolution grid
│   ├── fetch_osrm_matrix.py        # snap filter + chunked OSRM /table caller
│   ├── gen_offnetwork.py           # random raw coords -> OSRM-snapped labels (E5)
│   ├── build_road_density.py       # OSM road-density raster via pyosmium (E4)
│   ├── train_export_onnx.py        # LightGBM training, ONNX export, verification
│   ├── export_binary.py            # model.onnx -> model.bin (E11, no retraining)
│   ├── experiments.py              # balanced-eval experiment harness (E1–E4/E6/E7/E9)
│   └── experiment_e5.py            # off-network training experiment (E5)
├── server/                         # .NET 10 minimal API
│   ├── RoutingService.csproj
│   ├── Program.cs
│   ├── TreeEnsembleModel.cs        # E11/E11b: dependency-free interpreter, streams packed model.bin
│   └── models/
│       ├── model.onnx              # baked LightGBM ensemble (511 leaves, ONNX-verified)
│       ├── model.bin               # compiled tree tables served at runtime (E11)
│       └── model_metadata.json     # feature/target contract + held-out metrics
└── tests/
    ├── benchmark.py                # latency / RSS / accuracy + golden fixtures
    ├── osrm_vs_onnx.py             # standalone uv script: full accuracy statistics
    ├── OSRM_VS_ONNX.md             # generated statistics report (uniform sampling)
    ├── OSRM_VS_ONNX_STRATIFIED.md  # generated statistics report (band-balanced)
    ├── golden_routes.json          # checked-in benchmark output (test fixture)
    ├── RoutingService.Tests.csproj
    └── GoldenRoutesTests.cs
```

---

## Measured results

All numbers below were produced on the reference machine (16 vCPU, WSL2) by
`npm run bench` (`tests/benchmark.py`) and `npm run test`. The raw output is checked in as
`tests/golden_routes.json`, and the C# suite (`tests/GoldenRoutesTests.cs`) asserts both model
reproducibility and the accuracy budgets, so these are regression-gated rather than prose.

### Accuracy (training, held-out)

5,009,804 pairs (2,203 routable grid points + off-network samples); 10% held out.

| target | MAE | MedAE | MedAPE | RMSE |
|--------|-----|-------|--------|------|
| `distance_m` | 788.4 | 525.3 | 3.3% | 1,202.7 |
| `duration_s` | 36.5 | 29.1 | 2.3% | 48.9 |

The ONNX graph reproduces the LightGBM boosters to **100.0000%** on the held-out set
(max |Δ| = 0.03 m / 0.002 s). A haversine-as-distance baseline scores 29.1% MedAPE, so the
model is a large improvement over geometry alone. Error is smallest on long trips and largest
at short range, where OSRM follows the road network rather than the straight line
(`distance_m` MedAPE: 20.0% under 1 km → 1.2% over 25 km).

### Accuracy (served model vs **live** OSRM)

600 random coordinate pairs, deliberately sampled outside the training grid:

| target | MedAE | MedAPE | p90 APE | within 10% |
|--------|-------|--------|---------|------------|
| `duration_s` | 87.4 | 4.9% | 19.1% | 78% |
| `distance_m` | 1,185.4 | 4.1% | 16.0% | 77% |

### Accuracy statistics and the distance caveat

`tests/osrm_vs_onnx.py` (a self-contained uv script — `./tests/osrm_vs_onnx.py --n 5000`, or
`npm run stats`) does a fuller job: it samples random pairs, pulls exact ground truth from a live
OSRM `/table` in chunks, and reports MAE/RMSE, signed bias, the APE percentile distribution,
correlation, an OLS fit, bootstrap 95% CIs, and a breakdown by trip distance. Reports land in
`tests/OSRM_VS_ONNX.md` / `.json`.

**Uniform random sampling is misleading, and this is the important finding.** Averaging over the
whole bbox (N=5,000) gives an encouraging MedAPE of 5.0% (duration) / 4.3% (distance) — but that
number is dominated by long trips, because uniform sampling over a ~0.3°×0.77° box almost never
produces short ones. Sampling *evenly across separation bands* (`--stratify`, N=4,400) tells the
real story: MedAPE 9.9% / 10.3%, and a mean (MAPE) that blows up to 74% / 68% because short trips
have enormous *relative* error. Broken down by OSRM distance:

| OSRM distance | n | duration MedAPE | distance MedAPE | distance bias |
|---|---|---|---|---|
| < 1 km | 299 | 78.0% | 74.4% | +650 m |
| 1–3 km | 688 | 24.8% | 26.2% | +200 m |
| 3–10 km | 1,064 | 17.5% | 21.0% | −400 m |
| 10–25 km | 917 | 7.9% | 8.7% | −314 m |
| > 25 km | 1,432 | 4.2% | 3.5% | −493 m |

So accuracy is **strongly distance-dependent**: excellent (~4%) above 25 km, weak (tens of
percent) below ~3 km. This is inherent to a smooth coordinate function — a sub-kilometre trip is
a few road hops, and random raw coordinates often *snap* hundreds of metres before routing even
begins.

These numbers are already the *improved* model. `IMPROVEMENTS.md` documents the whole path: the
original 63-leaf model had a band-balanced MedAPE of 12.7% / 14.2% and a `< 1 km` bucket of
**216.9%**; inverse-frequency training weights (E1) removed the implicit ~1 km floor and
off-network training pairs (E5) roughly halved the error on the raw coordinates the API actually
receives. Two follow-up experiments asked whether the remaining short-trip error is *snapping*
(give the model the exact OSRM snap) or *barrier-forced detour* (static river/rail/motorway
rasters). Both are rejected on measurement: together they move off-network short-trip MedAPE from
~39% to ~34% and stop. The residual is genuine route-choice / network topology, not a modelling gap
that a better snapper or more features could close — see `IMPROVEMENTS.md` (Experiment 0 / 1).

### Latency

| measurement | p50 | p90 | p99 | p99.9 |
|-------------|-----|-----|-----|-------|
| server-side (`Server-Timing`) | 94 µs | 167 µs | **361 µs** | 1,109 µs |
| client round-trip (localhost) | 687 µs | 872 µs | 1,488 µs | 2,623 µs |

The plan's **p99 < 1 ms** target is met on server-side processing time with ~2.8× headroom.
Compiling the trees to C# (E11) pulled every percentile down versus the ONNX Runtime build
(same box, back to back: server-side p50 118 → 94 µs, p99 436 → 361 µs) — the interpreter
walks 400 shallow trees per target with no native-call or tensor overhead. Client round-trip
is higher only because it includes Python `httpx` + loopback HTTP overhead.

### Throughput — head-to-head vs OSRM

Identical load profile (`wrk -t8 -c64 -d15s`) run against the approximation service and a live
`osrm-routed`, on the same 16-vCPU box, back to back. The OSRM container had **no CPU limit**
(unlimited cores; its steady-state footprint is measured in the Memory section below), so this is a
like-for-like comparison:

| metric | OSRM (`osrm-routed`) | Approx service |
|--------|----------------------|----------------|
| requests/sec | 13,053 | **43,532** |
| p50 | 4.34 ms | 1.32 ms |
| p90 | 7.66 ms | 2.26 ms |
| p99 | 11.06 ms | **4.21 ms** |
| avg latency | 4.91 ms | 1.48 ms |
| requests served (15 s) | 196,711 | 655,220 |
| response size | 591 B | 40 B |
| socket read errors | 350 | 0 |

That is roughly **3.3× the throughput and ~2.6× lower p99**, with no dropped connections where OSRM
recorded 350. Two honest caveats: this is a *single repeated route* (steady-state single-route
throughput, not a distribution over the map — OSRM does not cache, so it is still a fair
direction), and OSRM's larger JSON payload accounts for part of its latency. OSRM remains exact
ground truth; the approximation buys speed and a far lighter deployment at ~4.9%/4.1% MedAPE on
random pairs.

### Memory — head-to-head vs OSRM (**plan target not met**)

Both services were warmed with 200 synthetic `A → B` requests, then sampled from
`/proc/<pid>/smaps_rollup` at steady state (three identical readings, OSRM read inside its
container). The approximation column was re-measured after the memory fixes below; *cold*
is sampled immediately after model load, before any request, and the OSRM column is the
original head-to-head (it could not be re-read from outside its container):

| metric | OSRM (`osrm-routed`) | Approx service (warm) | Approx service (cold) |
|--------|----------------------|-----------------------|-----------------------|
| **Pss** (shared-page adjusted) | **574 MB** | **79 MB** | 59 MB |
| RSS | 575 MB | 104 MB | 83 MB |
| anonymous heap (`Private_Dirty`) | 552 MB | 53 MB | — |
| peak high-water (`VmHWM`) | 710 MB | 102 MB | — |
| cgroup usage (container) | 558 MB | — (not containerised) | — |
| on-disk artifact | 208 MB (`.osrm*` MLD graph) | 14 MB (`model.bin`) | 14 MB |
| threads | 18 | 16 | 16 |

The approximation therefore sits **~495 MB below OSRM by Pss (~86% lower; 7.3× ratio)** and
~82% lower by RSS — the opposite of what the plan assumed. OSRM's anonymous heap balloons under the
208 MB MLD graph (552 MB anon); the approximation's 53 MB anon heap is now dominated by fixed
.NET/ASP.NET Core host overhead on top of a 14 MB tree table. One caveat on the ratio: OSRM's
footprint depends on its state (its `VmRSS` reads ~303 MB when idle in this session, against 575 MB
under the `wrk` load above), and only the loaded figure is comparable to a warmed service.

The two scale differently, which matters more than the single number: OSRM's footprint grows with
the road graph, so a larger region pushes it well past this, whereas the approximation is roughly
*flat* (fixed runtime overhead + model) — the gap widens with map size and would shrink or reverse
for a very small map. It also ships without a multi-hundred-MB graph.

**The plan's `< 30 MB` target is still not met**, but the gap is smaller than the previous revision
of this section claimed, and its cause is now measured rather than assumed. Compiling the trees to
C# and dropping ONNX Runtime (E11, q.v.) removed the native runtime *and* the ONNX tree node table
in one step: RSS **428 → 136 MB (−68%)**, Pss **391 → 99 MB**, anon heap **359 → 85 MB**, with
server-side latency improving at every percentile. What that analysis missed is that **E11 left the
model costing ~1.9× its own file size in RSS** — each MB of `model.bin` added ~1.86 MB resident —
because `TreeEnsembleModel.Load` read the file whole with `File.ReadAllBytes`, putting a second,
full-size copy of the table on the Large Object Heap on top of the ~18 MB of parsed node arrays.
That copy is now gone: the loader streams each array straight into its final allocation, and an A/B
on the same box drops the 511-leaf model's cost from **1.92× to 1.00×** its artifact size (cold RSS
104.5 → 87.0 MB) and warm RSS under load from **123 → 108 MB** — with predictions bit-identical
over 400 routes and all 38 tests passing. The table itself then narrowed from 22 to 17 bytes per
node (uint8 feature indices, leaf flag folded into the sign of the child index — see
`IMPROVEMENTS.md` E11b), taking `model.bin` from 17.97 to **13.89 MB** and a further **4 MB** off
RSS to the 104 MB above, still bit-identical. Readings are stable to <0.5 MB across runs, so this
is a footprint, not a leak. Approaching single-digit MB would still require a non-.NET host; that
trade-off remains out of scope. The serving *compute* target (p99 < 1 ms, zero graph traversal) is
met; the *footprint* target is not.

The 511-leaf capacity (E7) is what made the ONNX-era footprint heavy — loading the 63-leaf
`model.onnx` (3.8 MB) into the ONNX server gave **~138 MB RSS**, the 511-leaf one (32.4 MB) gave
**~400 MB**. E11 broke most of that coupling and the streaming load finished the job: the tree table
is read once into flat arrays and now costs **~1.0× its own file size** (~14 MB for the shipped
model), down from the ~1.9× it cost between E11 and the loader fix. The saving scales with the
model, so it is ~17 MB here and only ~3 MB for the old 2.2 MB 63-leaf table — capacity is no longer
an expensive lever for memory. A cross-library test confirmed the shape is already on the Pareto
front: at the same ~18 MB, XGBoost scores 7% worse and CatBoost 2× worse, and no cheaper LightGBM
configuration holds the current accuracy (`IMPROVEMENTS.md` E12).

---

## Implementation notes

A few things about OSRM's HTTP API are easy to get wrong and are handled explicitly here:

- **Coordinate order is `lon,lat`.** Passing `lat,lon` does not error — points snap to a
  degenerate location and the API silently returns `0 m / 0 s` matrices.
- **`/nearest` accepts exactly one coordinate per request** (`Only one input coordinate is
  supported`), so snapping fans out over a thread pool.
- **`annotations=duration,distance` must live in the URL query string.** A POST body is
  ignored; requesting `distances` that way yields a response with only `durations`.
- **`;` and `,` must not be percent-encoded.** OSRM splits on these characters before
  decoding, so `%3B` / `%2C` silently produce all-zero matrices.
- Grid points further than 250 m from the nearest road are dropped before sampling, since
  they snap to distant boundary edges and inject degenerate labels.

## Caveats

The model is a smooth function of coordinates. It cannot represent genuinely discontinuous
routing behaviour — a river with few crossings, one-way systems, turn restrictions — so
error grows where the road network is not locally uniform. Medium-range trips across the
city, where the network offers several distinct corridors, are the hardest case. Accuracy
numbers are reported honestly in the benchmark output.

Sub-kilometre trips from *raw* coordinates are the weakest point, and for a reason no coordinate
model can fix. Training on off-network pairs (`IMPROVEMENTS.md` E5) halved the error there but
cannot remove it. Two experiments tested the obvious explanations and both failed to close the
gap: handing the model the **exact** OSRM snap (Experiment 0) cuts off-network short-trip MedAPE
only 38.9% → 35.8% (distance) / 38.3% → 34.1% (duration), and adding static connectivity/detour
rasters (Experiment 1) reaches ~34% — still ~8× the on-network level. The residual is genuine
route-choice / topology error, not snap uncertainty or barrier geometry.

`IMPROVEMENTS.md` is the honest record of what worked and what did not: three adopted changes
(inverse-frequency weights, off-network data, compiled-tree serving) plus a capacity bump, and
six measured rejections — log/normalised targets, sin/cos bearing, road-density features,
short-range specialist, the exact snap, connectivity rasters — with the RSS/size cost of the
adopted configuration stated rather than hidden.
