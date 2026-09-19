# OSRM vs ONNX approximation — accuracy statistics

_Generated 2026-09-19T20:53:14.040711+00:00 by `tests/osrm_vs_onnx.py`._

- Sample: **4,987** uniformly random pairs (seed 7)
- Ground truth: live OSRM `/table` · Predictions: `GET /route` on http://localhost:5080
- Units: duration in seconds, distance in metres. APE = |pred − truth| / truth.

## Headline

| metric | duration | distance |
|---|---|---|
| MAE | 128.6 s | 1,758.7 m |
| MedAE | 89.9 s | 1,200.2 m |
| RMSE | 199.8 s | 2,731.3 m |
| MedAPE | 5.0% | 4.3% |
| MAPE | 11.5% | 10.5% |
| within 10% | 76% | 78% |
| within 25% | 95% | 94% |
| Pearson r | 0.9621 | 0.9809 |
| bias | -0.5 s | +125.1 m |

## Detail

### Duration

- **n** = 4,987 pairs (all routable)
- **MAE** = 128.6 s  (95% CI 124.5 – 133.1)
- **MedAE** = 89.9 s
- **RMSE** = 199.8 s
- **MedAPE** = 5.0%  (95% CI 4.8 – 5.1)
- **MAPE** (mean) = 11.5%
- **signed bias** = -0.5 s (-0.0% of mean truth; 95% CI -6.1 – +5.1)
- **APE percentiles** = p50 5.0% · p75 9.7% · p90 17.5% · p95 25.5% · p99 74.9%
- **within** 5% 50% · 10% 76% · 20% 92% · 25% 95%
- **correlation** Pearson r = 0.9621, Spearman ρ = 0.9642
- **OLS fit** pred ≈ 133.1 + 0.9296·truth ((slight under-prediction))

### Distance

- **n** = 4,987 pairs (all routable)
- **MAE** = 1,758.7 m  (95% CI 1,700.9 – 1,819.2)
- **MedAE** = 1,200.2 m
- **RMSE** = 2,731.3 m
- **MedAPE** = 4.3%  (95% CI 4.1 – 4.4)
- **MAPE** (mean) = 10.5%
- **signed bias** = +125.1 m (+0.4% of mean truth; 95% CI +51.8 – +200.0)
- **APE percentiles** = p50 4.3% · p75 8.9% · p90 17.9% · p95 27.4% · p99 80.6%
- **within** 5% 56% · 10% 78% · 20% 91% · 25% 94%
- **correlation** Pearson r = 0.9809, Spearman ρ = 0.9804
- **OLS fit** pred ≈ 1214.2 + 0.9641·truth ((slight over-prediction))

## Stratified by OSRM distance

#### Duration

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| 1-3 km | 40 | 130.1 | 83.6 | 223.1 | +95.0 | 41.8% | 199.6% | 20% |
| 3-10 km | 372 | 152.9 | 84.7 | 247.0 | +84.6 | 13.7% | 65.3% | 42% |
| 10-25 km | 1,414 | 121.2 | 90.0 | 173.6 | +13.1 | 6.5% | 19.4% | 65% |
| >25 km | 3,155 | 128.5 | 90.2 | 202.0 | -18.7 | 4.1% | 11.8% | 85% |

#### Distance

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| 1-3 km | 40 | 1,562.8 | 608.0 | 3,446.9 | +1,266.6 | 29.2% | 159.7% | 25% |
| 3-10 km | 372 | 1,933.0 | 922.8 | 3,588.1 | +1,240.0 | 14.4% | 68.9% | 39% |
| 10-25 km | 1,414 | 1,782.7 | 1,245.2 | 2,666.1 | +607.3 | 6.9% | 23.0% | 65% |
| >25 km | 3,155 | 1,721.1 | 1,220.0 | 2,589.9 | -248.7 | 3.2% | 10.2% | 90% |

