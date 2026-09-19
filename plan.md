# Plan — Fast OSRM Approximation Service (.NET 10 + ONNX)

**Goal:** Provide an ultra-fast `A → B` lookup returning static `(duration_s, distance_m)` mimicking OSRM routing in Greater Manchester, with **p99 < 1 ms, single-digit MB RSS**, zero per-query graph traversal, and no runtime OSRM container needed.

---

## 1. Architecture Overview

- **Offline / Training (Python):**
  1. Extract Greater Manchester OSM data and run OSRM backend container (`osrm-extract`, `osrm-partition`, `osrm-customize`).
  2. Sample grid points across Manchester (dense 100m–250m central, 500m–1km outer).
  3. Query OSRM `/table` API in chunks to produce ground-truth static `(duration_s, distance_m)`.
  4. Train LightGBM regression models:
     - Features: `orig_lat`, `orig_lon`, `dest_lat`, `dest_lon`, `haversine_dist_m`, `bearing_deg`, `lat_delta`, `lon_delta`.
     - Target: `osrm_distance_m` and `osrm_duration_s`.
  5. Export trained models to ONNX (`model.onnx`).

- **Online / Serving (.NET 10 Minimal API):**
  1. Standalone C# .NET 10 service running `Microsoft.ML.OnnxRuntime`.
  2. On request `GET /route?orig=lat,lon&dest=lat,lon`:
     - Compute geometric features (haversine, bearing, deltas) in microsecond C# math.
     - Single-pass ONNX inference.
     - Return `{ duration_s, distance_m }`.
  3. Latency: < 1 ms; Memory: ~10–25 MB RSS. OSRM is offline and never queried at runtime.

---

## 2. Directory Structure

```
testing-ml-tte/
├── plan.md
├── docker-compose.yml              # OSRM backend for Manchester map (offline generation only)
├── package.json                    # simple runner scripts (npm run data / train / serve / bench)
├── data/
│   ├── raw/                        # greater-manchester.osm.pbf
│   └── processed/                  # samples.parquet (coords, distance, duration)
├── python/                         # self-contained uv scripts (PEP 723 inline metadata)
│   ├── generate_grid.py            # multi-resolution grid generator
│   ├── fetch_osrm_matrix.py        # chunked OSRM /table caller
│   └── train_export_onnx.py        # LightGBM training & ONNX export
├── server/                         # .NET 10 Minimal API
│   ├── RoutingService.csproj       # .NET 10, Microsoft.ML.OnnxRuntime
│   ├── Program.cs                  # GET /route endpoint + feature prep + ONNX inference
│   └── models/
│       └── model.onnx              # Baked lightweight model
└── tests/
    ├── benchmark.py                # Latency & accuracy comparison vs live OSRM
    └── GoldenRoutesTests.cs        # Unit & regression tests
```

---

## 3. Milestones

1. **M0: OSRM Manchester Setup** — Download Manchester extract, build OSRM car profile, verify `/table` and `/route`.
2. **M1: Dataset Generation** — Generate multi-resolution grid, extract pairwise distance & duration via `/table` chunks into Parquet.
3. **M2: Model Training & ONNX Export** — Train LightGBM models, evaluate MedAE & MAE against exact OSRM, export to `model.onnx`.
4. **M3: .NET 10 Serving Service** — Implement ASP.NET Core minimal API with embedded ONNX runtime.
5. **M4: Benchmarking & Decommissioning** — Measure latency (< 1 ms p99), RSS (< 30 MB), verify accuracy, and shut down OSRM.
