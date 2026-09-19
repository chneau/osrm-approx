# OSRM vs ONNX approximation — accuracy statistics

_Generated 2026-09-19T20:12:44.686470+00:00 by `tests/osrm_vs_onnx.py`._

- Sample: **4,987** uniformly random pairs (seed 7)
- Ground truth: live OSRM `/table` · Predictions: `GET /route` on http://localhost:5080
- Units: duration in seconds, distance in metres. APE = |pred − truth| / truth.

## Headline

| metric | duration | distance |
|---|---|---|
| MAE | 190.3 s | 2,996.2 m |
| MedAE | 120.6 s | 1,819.4 m |
| RMSE | 288.2 s | 4,620.6 m |
| MedAPE | 6.8% | 6.4% |
| MAPE | 17.9% | 17.6% |
| within 10% | 64% | 64% |
| within 25% | 88% | 87% |
| Pearson r | 0.9232 | 0.9484 |
| bias | +50.5 s | +1,145.6 m |

## Detail

### Duration

- **n** = 4,987 pairs (all routable)
- **MAE** = 190.3 s  (95% CI 184.6 – 196.7)
- **MedAE** = 120.6 s
- **RMSE** = 288.2 s
- **MedAPE** = 6.8%  (95% CI 6.6 – 7.0)
- **MAPE** (mean) = 17.9%
- **signed bias** = +50.5 s (+2.7% of mean truth; 95% CI +42.7 – +58.4)
- **APE percentiles** = p50 6.8% · p75 13.9% · p90 27.6% · p95 43.8% · p99 131.5%
- **within** 5% 40% · 10% 64% · 20% 84% · 25% 88%
- **correlation** Pearson r = 0.9232, Spearman ρ = 0.9197
- **OLS fit** pred ≈ 249.3 + 0.8953·truth ((slight over-prediction))

### Distance

- **n** = 4,987 pairs (all routable)
- **MAE** = 2,996.2 m  (95% CI 2,896.6 – 3,100.4)
- **MedAE** = 1,819.4 m
- **RMSE** = 4,620.6 m
- **MedAPE** = 6.4%  (95% CI 6.2 – 6.7)
- **MAPE** (mean) = 17.6%
- **signed bias** = +1,145.6 m (+3.8% of mean truth; 95% CI +1,023.7 – +1,269.9)
- **APE percentiles** = p50 6.4% · p75 14.9% · p90 32.4% · p95 52.0% · p99 169.6%
- **within** 5% 41% · 10% 64% · 20% 82% · 25% 87%
- **correlation** Pearson r = 0.9484, Spearman ρ = 0.9455
- **OLS fit** pred ≈ 3178.6 + 0.9330·truth ((slight over-prediction))

## Stratified by OSRM distance

#### Duration

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| 1-3 km | 40 | 254.0 | 125.8 | 412.4 | +240.3 | 47.9% | 457.7% | 18% |
| 3-10 km | 372 | 216.6 | 109.9 | 341.4 | +161.5 | 17.7% | 118.6% | 34% |
| 10-25 km | 1,414 | 188.8 | 118.1 | 286.5 | +78.2 | 9.0% | 31.0% | 54% |
| >25 km | 3,155 | 186.5 | 123.2 | 279.2 | +21.7 | 5.4% | 19.2% | 73% |

#### Distance

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| 1-3 km | 40 | 3,308.0 | 1,162.4 | 6,112.2 | +3,254.8 | 57.7% | 573.8% | 12% |
| 3-10 km | 372 | 3,131.2 | 1,409.5 | 5,313.1 | +2,715.9 | 19.9% | 159.9% | 25% |
| 10-25 km | 1,414 | 3,183.1 | 1,799.2 | 5,125.0 | +2,069.5 | 10.2% | 46.0% | 49% |
| >25 km | 3,155 | 2,885.3 | 1,870.7 | 4,248.7 | +508.9 | 4.9% | 19.3% | 76% |

