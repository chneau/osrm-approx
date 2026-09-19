# OSRM vs ONNX approximation — accuracy statistics

_Generated 2026-09-19T20:53:39.227324+00:00 by `tests/osrm_vs_onnx.py`._

- Sample: **4,400** stratified-across-bands pairs (seed 7)
- Ground truth: live OSRM `/table` · Predictions: `GET /route` on http://localhost:5080
- Units: duration in seconds, distance in metres. APE = |pred − truth| / truth.

## Headline

| metric | duration | distance |
|---|---|---|
| MAE | 143.2 s | 1,826.8 m |
| MedAE | 84.5 s | 1,013.8 m |
| RMSE | 254.7 s | 3,354.4 m |
| MedAPE | 9.9% | 10.3% |
| MAPE | 74.4% | 68.3% |
| within 10% | 50% | 49% |
| within 25% | 74% | 72% |
| Pearson r | 0.9612 | 0.9789 |
| bias | -25.6 s | -247.1 m |

## Detail

### Duration

- **n** = 4,400 pairs (all routable)
- **MAE** = 143.2 s  (95% CI 137.0 – 149.5)
- **MedAE** = 84.5 s
- **RMSE** = 254.7 s
- **MedAPE** = 9.9%  (95% CI 9.3 – 10.3)
- **MAPE** (mean) = 74.4%
- **signed bias** = -25.6 s (-2.1% of mean truth; 95% CI -33.0 – -18.4)
- **APE percentiles** = p50 9.9% · p75 26.7% · p90 61.0% · p95 90.7% · p99 535.4%
- **within** 5% 32% · 10% 50% · 20% 69% · 25% 74%
- **correlation** Pearson r = 0.9612, Spearman ρ = 0.9493
- **OLS fit** pred ≈ 23.4 + 0.9591·truth ((slight under-prediction))

### Distance

- **n** = 4,400 pairs (all routable)
- **MAE** = 1,826.8 m  (95% CI 1,743.4 – 1,909.3)
- **MedAE** = 1,013.8 m
- **RMSE** = 3,354.4 m
- **MedAPE** = 10.3%  (95% CI 9.6 – 11.0)
- **MAPE** (mean) = 68.3%
- **signed bias** = -247.1 m (-1.4% of mean truth; 95% CI -342.5 – -154.4)
- **APE percentiles** = p50 10.3% · p75 28.6% · p90 63.7% · p95 91.1% · p99 634.0%
- **within** 5% 33% · 10% 49% · 20% 67% · 25% 72%
- **correlation** Pearson r = 0.9789, Spearman ρ = 0.9624
- **OLS fit** pred ≈ -41.6 + 0.9884·truth ((slight under-prediction))

## Stratified by OSRM distance

#### Duration

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| <1 km | 299 | 85.0 | 51.6 | 125.5 | +73.1 | 78.0% | 787.9% | 10% |
| 1-3 km | 688 | 83.8 | 56.0 | 143.1 | +11.4 | 24.8% | 71.7% | 21% |
| 3-10 km | 1,064 | 161.0 | 99.8 | 245.9 | -50.4 | 17.5% | 65.0% | 31% |
| 10-25 km | 917 | 166.7 | 95.4 | 313.0 | -37.2 | 7.9% | 31.0% | 60% |
| >25 km | 1,432 | 155.5 | 93.6 | 280.4 | -38.1 | 4.2% | 13.4% | 81% |

#### Distance

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| <1 km | 299 | 723.0 | 361.4 | 1,300.2 | +650.2 | 74.4% | 850.5% | 8% |
| 1-3 km | 688 | 777.3 | 446.9 | 1,709.3 | +200.1 | 26.2% | 71.3% | 20% |
| 3-10 km | 1,064 | 1,843.7 | 1,154.6 | 2,863.5 | -400.3 | 21.0% | 70.2% | 27% |
| 10-25 km | 917 | 2,282.7 | 1,406.7 | 3,665.6 | -314.0 | 8.7% | 33.7% | 54% |
| >25 km | 1,432 | 2,256.9 | 1,323.2 | 4,256.7 | -492.7 | 3.5% | 12.9% | 86% |

