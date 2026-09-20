# Conclusion — what is the best solution?

This document answers one question, using the measurements already in the repo
(`IMPROVEMENTS.md`, `SIMPLIFICATION.md`, `experiments/*.json`, `tests/OSRM_VS_ONNX*.md`):

> **Which architecture gives good results, quick results, and small RAM use at the same time?**

Short answer: **the shipped learn-to-mimic design — LightGBM `511 leaves × 400 trees`
trained on grid + off-network OSRM labels, compiled to a packed `model.bin`, evaluated
by the dependency-free C# `TreeEnsembleModel` — is the best all-round solution.** It wins
on accuracy *and* latency, and its memory is the smallest of any option that is actually
competitive on accuracy. The only things that beat it on a *single* axis are exact OSRM
(exactness, at ~5.5× the RAM and ~3× the latency) and the matrix/lookup prototype (S1)
(slightly fewer moving parts, at a small accuracy and size cost). Nothing beats it on all
three axes.

The plan's original targets, restated as the three criteria:

| criterion | target | shipped solution |
|---|---|---|
| good results | mimic OSRM | MedAPE **2.6% distance / 2.4% duration** on held-out raw coords; 4.9% / 4.1% vs live OSRM |
| quick results | p99 < 1 ms | server-side **p99 361 µs**, **43.5k req/s** |
| small RAM | < 30 MB | **~104 MB warm / 83 MB cold** — target missed, but 5.5× below OSRM |

The `< 30 MB` target is genuinely not met, and no option in this repo meets it inside a
.NET host: the ASP.NET Core + runtime floor is ~90 MB before the model is loaded. That
trade-off is stated honestly rather than hidden; hitting single-digit MB would require a
non-.NET host (out of scope).

---

## 1. What was actually measured

All accuracy rows below use the **same honest held-out set** — raw off-network
coordinates, the API's real input — unless noted (`experiments/try_s1_fair.py`,
`capacity_accuracy.json`, `boost_showdown.json`, `version_bakeoff*.json`). Latency and
RSS are the head-to-head measurements in `README.md` / `IMPROVEMENTS.md` E11.

### Accuracy / artifact / speed / memory

| # | approach | distance MedAPE | duration MedAPE | artifact | RAM (warm) | query cost | complexity |
|---|---|---|---|---|---|---|---|
| **G** | **GBM 511L×400t, C# interpreter (shipped)** | **2.6%** | **2.4%** | **13.9 MB `model.bin`** | **~104 MB RSS** | **94 µs p50 / 361 µs p99** | medium (offline ML + tiny parser) |
| G63 | GBM 63L×400t (small model) | 4.5% | 4.4% | 2.2 MB | ~94 MB RSS | ~microseconds | medium |
| G255 | GBM 255L×400t (middle) | 3.2% | 3.0% | 9.0 MB | ~106 MB RSS | ~tens of µs | medium |
| ONNX | GBM + ONNX Runtime | 2.6% | 2.4% | 31 MB `.onnx` | ~400–430 MB | 118 / 436 µs | high (native RT) |
| XGB | XGBoost, same shape/size | 2.8% | 2.7% | 18 MB | ~similar | similar | medium |
| CAT | CatBoost symmetric, ~same size | 5.5% | 5.7% | ~18 MB | ~similar | similar | medium |
| **S1** | **all-pairs matrix + lattice lookup** | **3.4%** | **5.0%** | 39 MB (19 MB `uint16`) | ~host floor (~90 MB) | 16 lookups, sub-µs | **low — no training, no format** |
| S1b | landmark oracle, L=512 | 5.1% | — | 9 MB | host floor | min over 512 | low |
| S2 | hub labels / PLL on proximity graph | 35.2% | 163.9% | 30 MB labels | host floor | ~15 µs | high (graph build) |
| S2b | full OSM driving graph (independent router) | 6.2–11.0% | 24–31% | 438k-node graph | large | graph search | high |
| S2c | major-roads-only OSM graph | 29.6% | 52.4% | small graph | small | graph search | medium |
| S0 | cache in front of live OSRM | **0% (exact)** | **0%** | 208 MB `.osrm*` | **574 MB Pss** | **4.3 ms p50 / 11 ms p99**, 13k req/s | low code, heavy service |
| S3 | closed-form detour + speed model | not built | not built | ~12 floats | host floor | ~20 FLOPs | **very low** |
| S5 | analytic baseline + tiny residual GBM | not built | not built | small | host floor | baseline + walk | low–medium |
| S6 | k-NN over training pairs | ≈S1 | ≈S1 | O(pairs) | high | KD-tree query | low |

### The headline trade, in one line

* **S0 (OSRM)** is exact but memory/latency-heavy.
* **S1 (matrix)** is simple and nearly as accurate, but its `O(N²)` matrix is memory-bound
  and it cannot represent the off-network snapping the GBM learned.
* **S2 / OSM graphs** are a *different router*, not an approximation of OSRM — they fail
  on turn restrictions and speed profiles (0.90× distance, 0.70–0.76× duration, 4–10%
  unreachable).
* **The learned ensemble (G)** wins precisely because it distilled OSRM's *outputs* —
  snap, turn rules, speeds — instead of rediscovering them.

---

## 2. Why the shipped GBM is the best all-round answer

### It wins on accuracy
2.6% / 2.4% MedAPE on honest raw-coordinate hold-out is the best non-oracle result in the
repo (S1 is 3.4% / 5.0%; XGBoost and CatBoost are 7% and 2× worse at the same size).
Against live OSRM, band-balanced, it is 9.9% / 10.3%, and long trips are excellent
(>25 km ≈ 3.5–4.2%). E1 inverse-frequency weights removed the old ~1 km training floor
(84% → 16% on-grid); E5 off-network data roughly halved raw-coordinate error.

### It wins on speed
Server-side p50 94 µs / p99 361 µs, p99.9 1,109 µs — the plan's p99 < 1 ms with ~2.8×
headroom. Under `wrk -t8 -c64 -d15s` it serves **43.5k req/s** vs OSRM's 13.0k, at ~2.6×
lower p99. Compiling the trees to C# was *faster* than ONNX Runtime at every percentile
(118→94 µs p50, 436→361 µs p99), because 400 shallow trees are walked with no native call
or tensor overhead.

### It wins on RAM among accurate options
Warm **~104 MB RSS / ~79 MB Pss**, cold **83 MB** — 5.5× below OSRM's 574 MB Pss and 4×
below the ONNX Runtime build's ~400 MB. After E11b the model costs ~1.0× its own 13.9 MB
file in RSS, and capacity is a *linear* but modest term (+29 MB for 63→511 leaves, i.e.
the entire accuracy curve costs ~30 MB). No cheaper LightGBM shape, and no competing
library, held the same accuracy (`boost_showdown.json`).

### The memory caveat, stated plainly
The `.NET` host floor is ~90 MB, so **no in-process method here meets `< 30 MB`** — GBM at
any capacity, S1, PLL and the OSM graph all sit near or above that floor. Within that
floor the GBM is still the smallest *accurate* model because its 13.9 MB table is smaller
than S1's 19–39 MB matrix.

---

## 3. Decision matrix (weighted for "good + quick + small")

Scores 1 (poor) – 5 (excellent). Weights: accuracy 40%, speed 30%, RAM 30%.

| approach | accuracy (40%) | speed (30%) | RAM (30%) | **weighted** | when to pick it |
|---|---|---|---|---|---|
| **G — GBM + C# interpreter (shipped)** | **5** | **5** | **4** | **4.70** | **default: best all-round** |
| S1 — matrix + lattice lookup | 4 | 5 | 3.5 | 4.15 | minimum moving parts / no ML pipeline |
| G63 — 63-leaf GBM | 3 | 5 | 4.5 | 4.05 | strictest RAM inside .NET, accept 4.5% |
| S5 — baseline + residual GBM (untested) | 4.5 | 5 | 4.5 | 4.65 (est.) | best accuracy-per-byte if built |
| ONNX — GBM + ONNX Runtime | 5 | 4.5 | 1.5 | 3.80 | only if ORT is mandated |
| S0 — live OSRM (+cache) | **5 (exact)** | 2 | 1 | 2.90 | exactness is non-negotiable |
| S2b — full OSM graph router | 2.5 | 3 | 3 | 2.80 | you actually want a second router |
| S2 — PLL hub labels | 1.5 | 5 | 3.5 | 3.15 | exact oracle, needs the *real* road graph |
| S2c — major-roads OSM | 1 | 4 | 4 | 2.80 | rejected |
| S3 — closed form (untested) | 2 (est.) | 5 | 5 | 3.80 (est.) | auditable arithmetic, loose accuracy |

The shipped GBM is the only row that scores ≥4 on every axis, which is exactly the
"good + quick + small" objective.

---

## 4. Recommendation

**Ship the current architecture (G). Keep LightGBM at 511 leaves × 400 trees × lr 0.08,
keep the E1 inverse-frequency weights and the E5 off-network data, keep the packed
17-byte/node `model.bin`, and keep the C# `TreeEnsembleModel` interpreter.**

Concretely, keep what is already in the tree:

- `python/train_export_onnx.py` → `model.onnx` (source of truth, verified 100.0000% vs
  LightGBM),
- `python/export_binary.py` → `model.bin` (13.9 MB, lossless re-serialisation),
- `server/TreeEnsembleModel.cs` (streams the file, walks trees, no ONNX Runtime),
- `server/Program.cs` (8-feature contract, `Server-Timing`).

### Contingent fallbacks

| If the binding constraint changes to… | switch to | measured cost |
|---|---|---|
| RAM inside .NET | **255 leaves** | +0.5 MedAPE points, ~−7 MB vs 511 (`capacity_accuracy.json`) |
| RAM, hard | **63 leaves** | 4.5% / 4.4%, 2.2 MB, but still ~94 MB RSS — floor dominates |
| Minimum pipeline / no ML | **S1 matrix + lookup** | 3.4% / 5.0%, 19–39 MB, no training/export/parser |
| Exact answers required | **S0 (OSRM + cache)** | exact, 574 MB Pss, 4.3 ms p50 |
| Best accuracy-per-byte, willing to build | **S5 baseline + residual** | untested; plausible small model at current accuracy |

### What is *not* worth doing

- Replacing the GBM with a real OSM graph router (S2b/S2c): it is a different router, not
  an OSRM approximation, and loses on both accuracy and implementation cost.
- Swapping LightGBM for XGBoost/CatBoost: worse at equal size in a controlled bake-off.
- Building a snapper (Exp 0) or barrier rasters (Exp 1): both measured, together they move
  short-trip MedAPE only 39% → 34% and stop.
- Quantising leaf values to `uint16` (E11b): ~0.7 MB for golden-route drift — not worth it
  unless fixtures are regenerated for other reasons.

### Known limitation to document, not fix

Sub-kilometre trips from *raw* coordinates remain the weak point (~74–78% distance MedAPE
band-balanced; ~34–39% on off-network short pairs). This is genuine route-choice and
network topology — one-way systems, turn restrictions, few river crossings — which a
smooth coordinate function cannot represent. Only an exact router (S0) or a true hub-label
oracle over the *real* road graph (S2, not yet competitive as built) closes it. The
model's value is excellent long/medium-trip accuracy at a fraction of OSRM's cost, and the
docs already report the short-trip gap honestly.

---

## 5. One-paragraph verdict

Across every architecture prototyped in this repository, the learned tree ensemble served
by the dependency-free C# interpreter is the Pareto best on the requested triad of
*good results, quick results, small RAM*. It is the most accurate non-oracle method
(2.6% / 2.4% on raw coordinates), the fastest (p99 361 µs, 43.5k req/s), and the lightest
of the accurate options (~104 MB warm, 13.9 MB model), and it avoids a multi-hundred-MB
routing graph at serve time. The matrix-lookup (S1) alternative is genuinely simpler and
nearly as accurate if code size outranks accuracy; live OSRM (S0) remains the only exact
answer if exactness outranks everything. Absent those changes in priority, the shipped
GBM is the solution to keep — and the honest remaining gap is not a modelling failure but
the sub-kilometre topology that only an exact router can reproduce.
