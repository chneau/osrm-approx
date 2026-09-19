# Fast OSRM Approximation Service — Greater Manchester

An ultra-fast `A → B` lookup that returns static `(duration_s, distance_m)` mimicking OSRM
car routing in Greater Manchester — without running OSRM at query time.

OSRM is used **offline only** to generate ground truth. The runtime is a small .NET 10
service that computes a handful of geometric features and runs a single ONNX inference
pass. There is no graph traversal, no routing engine and no external dependency on the
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
multi-resolution grid (3000 pts)                  ONNX TreeEnsemble (1 pass)
   • 278 m core / 557 m mid / 1113 m fringe              ├── distance_m
        │                                                └── duration_s
        ▼                                                        │
OSRM /table (chunked) ──► samples.parquet                        ▼
        │                                         {"duration_s":..,"distance_m":..}
        ▼
LightGBM ×2 ──► model.onnx
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
`TreeEnsembleRegressor` nodes and share one graph, so serving is a single inference call.
LightGBM's boost-from-average initial score is measured empirically and baked in as each
node's `base_values`.

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

# 3. M2 — train LightGBM and export server/models/model.onnx
npm run train

# 4. M3 — serve
npm run serve           # listens on http://localhost:5080

# 5. M4 — benchmark latency / RSS / accuracy against live OSRM
npm run bench           # also writes tests/golden_routes.json
npm run test

# 6. Decommission OSRM
npm run osrm:down
```

### API

```bash
curl "http://localhost:5080/route?orig=53.4808,-2.2426&dest=53.3537,-2.2749"
# {"duration_s":1412.8,"distance_m":19243.8}
```

`GET /health` reports the loaded model and the feature contract. Every response carries a
`Server-Timing: app;dur=<ms>` header so server-side processing time can be measured without
the HTTP client's own overhead dominating the number.

---

## Repository layout

```
├── docker-compose.yml              # OSRM MLD build + serving (offline generation only)
├── package.json                    # runner scripts
├── data/
│   ├── raw/                        # greater-manchester.osm.pbf, .osrm* artefacts
│   └── processed/                  # grid.parquet, samples.parquet
├── python/                         # uv-managed project
│   ├── generate_grid.py            # multi-resolution grid
│   ├── fetch_osrm_matrix.py        # snap filter + chunked OSRM /table caller
│   └── train_export_onnx.py        # LightGBM training, ONNX export, verification
├── server/                         # .NET 10 minimal API
│   ├── RoutingService.csproj
│   ├── Program.cs
│   └── models/
│       ├── model.onnx              # baked LightGBM ensemble
│       └── model_metadata.json     # feature/target contract + held-out metrics
└── tests/
    ├── benchmark.py                # latency / RSS / accuracy + golden fixtures
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

4,850,998 pairs (2,203 routable grid points); 10% held out.

| target | MAE | MedAE | MedAPE | RMSE |
|--------|-----|-------|--------|------|
| `distance_m` | 1,280.1 | 923.7 | 5.9% | 1,808.1 |
| `duration_s` | 61.8 | 49.9 | 4.0% | 80.1 |

A haversine-as-distance baseline scores 29.2% MedAPE, so the model is a large improvement over
geometry alone. Error is smallest on long trips and largest at short range, where OSRM follows
the road network rather than the straight line (`distance_m` MedAPE: 30.7% under 1 km → 2.1%
over 25 km).

### Accuracy (served model vs **live** OSRM)

600 random coordinate pairs, deliberately sampled outside the training grid:

| target | MedAE | MedAPE | p90 APE | within 10% |
|--------|-------|--------|---------|------------|
| `duration_s` | 114.6 | 6.6% | 31.1% | 64% |
| `distance_m` | 1,833.8 | 6.7% | 30.3% | 64% |

### Latency

| measurement | p50 | p90 | p99 | p99.9 |
|-------------|-----|-----|-----|-------|
| server-side (`Server-Timing`) | 104 µs | 154 µs | **240 µs** | 598 µs |
| client round-trip (localhost) | 864 µs | 1,080 µs | 1,728 µs | 2,913 µs |

The plan's **p99 < 1 ms** target is met on server-side processing time with roughly 4× headroom.
Client round-trip is higher only because it includes Python `requests` + loopback HTTP overhead.

### Throughput

`wrk -t8 -c64 -d15s`: **46,380 req/s**, p99 4.31 ms. The service sustains well above the
sequential rate (≈1,100 req/s). At this offered load the measured p99 rises to ~4 ms; that
figure includes client and connection-handling cost at 64 concurrent connections, so it is an
upper bound on true per-request service latency rather than the server-side p99 above.

### Memory — **target not met**

Resident set size after warm-up is **~162 MB** (Δ ≈ +3 MB under load), against the plan's
**< 30 MB** target. This is not a code leak: the floor is the .NET/ASP.NET Core runtime plus the
native ONNX Runtime shared library, both of which the design depends on. A
`Process`-per-request or interpreter-free design would be required to approach single-digit MB;
that trade-off was out of scope. The serving *compute* target (p99 < 1 ms, no graph traversal) is
met, the *footprint* target is not.

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
