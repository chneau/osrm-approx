# OSRM vs ONNX approximation — accuracy statistics

_Generated 2026-09-19T20:13:13.112659+00:00 by `tests/osrm_vs_onnx.py`._

- Sample: **4,682** stratified-across-bands pairs (seed 7)
- Ground truth: live OSRM `/table` · Predictions: `GET /route` on http://localhost:5080
- Units: duration in seconds, distance in metres. APE = |pred − truth| / truth.

## Headline

| metric | duration | distance |
|---|---|---|
| MAE | 192.1 s | 2,645.3 m |
| MedAE | 113.8 s | 1,392.5 m |
| RMSE | 315.5 s | 4,463.4 m |
| MedAPE | 12.7% | 14.2% |
| MAPE | 130.1% | 134.9% |
| within 10% | 43% | 41% |
| within 25% | 67% | 65% |
| Pearson r | 0.9399 | 0.9634 |
| bias | +24.0 s | +615.7 m |

## Detail

### Duration

- **n** = 4,682 pairs (all routable)
- **MAE** = 192.1 s  (95% CI 184.7 – 199.4)
- **MedAE** = 113.8 s
- **RMSE** = 315.5 s
- **MedAPE** = 12.7%  (95% CI 12.0 – 13.6)
- **MAPE** (mean) = 130.1%
- **signed bias** = +24.0 s (+2.0% of mean truth; 95% CI +15.0 – +33.1)
- **APE percentiles** = p50 12.7% · p75 35.6% · p90 83.4% · p95 196.7% · p99 1053.7%
- **within** 5% 26% · 10% 43% · 20% 62% · 25% 67%
- **correlation** Pearson r = 0.9399, Spearman ρ = 0.9236
- **OLS fit** pred ≈ 106.6 + 0.9310·truth ((slight over-prediction))

### Distance

- **n** = 4,682 pairs (all routable)
- **MAE** = 2,645.3 m  (95% CI 2,546.9 – 2,749.8)
- **MedAE** = 1,392.5 m
- **RMSE** = 4,463.4 m
- **MedAPE** = 14.2%  (95% CI 13.5 – 15.1)
- **MAPE** (mean) = 134.9%
- **signed bias** = +615.7 m (+3.5% of mean truth; 95% CI +495.4 – +747.5)
- **APE percentiles** = p50 14.2% · p75 38.8% · p90 97.6% · p95 227.3% · p99 1309.6%
- **within** 5% 25% · 10% 41% · 20% 59% · 25% 65%
- **correlation** Pearson r = 0.9634, Spearman ρ = 0.9438
- **OLS fit** pred ≈ 1015.1 + 0.9775·truth ((slight over-prediction))

## Stratified by OSRM distance

#### Duration

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| <1 km | 324 | 192.6 | 129.7 | 282.2 | +186.7 | 176.5% | 1857.7% | 4% |
| 1-3 km | 719 | 133.5 | 77.3 | 235.1 | +87.9 | 34.5% | 155.9% | 17% |
| 3-10 km | 1,154 | 182.3 | 114.8 | 284.4 | +3.8 | 20.4% | 69.5% | 29% |
| 10-25 km | 962 | 217.2 | 122.0 | 370.7 | +3.4 | 9.9% | 42.9% | 50% |
| >25 km | 1,523 | 211.3 | 130.8 | 338.9 | -12.4 | 5.6% | 20.3% | 71% |

#### Distance

| bucket | n | MAE | MedAE | RMSE | bias | MedAPE | p90 APE | within 10% |
|---|---|---|---|---|---|---|---|---|
| <1 km | 324 | 1,985.5 | 1,183.1 | 3,535.7 | +1,962.4 | 216.9% | 1865.9% | 1% |
| 1-3 km | 719 | 1,574.7 | 697.7 | 3,409.9 | +1,319.1 | 38.5% | 154.1% | 15% |
| 3-10 km | 1,154 | 2,124.5 | 1,212.2 | 3,584.8 | +386.3 | 21.0% | 69.8% | 24% |
| 10-25 km | 962 | 3,202.7 | 1,769.9 | 5,021.8 | +549.9 | 11.4% | 52.0% | 45% |
| >25 km | 1,523 | 3,333.8 | 1,964.0 | 5,237.3 | +212.6 | 5.0% | 22.4% | 71% |

