# Experiment results

train=2,000,000 · balanced test=20,000 · leaves=63 trees=400 lr=0.08

| variant | features | target | weights | dist MedAPE | dur MedAPE | <1km | 1-3km | 3-10km | 10-25km | >25km | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | base | raw | none | 10.3% | 7.3% | 84% | 16% | 9% | 7% | 3% | 32 |
| E1_invfreq | base | raw | inv_freq | 7.9% | 6.4% | 16% | 12% | 8% | 8% | 3% | 23 |
| E2_log | base | log | none | 9.0% | 6.7% | 39% | 13% | 7% | 7% | 4% | 22 |
| E2_norm | base | norm | none | 9.6% | 7.5% | 41% | 14% | 8% | 8% | 4% | 20 |
| E3_sincos | sincos | raw | none | 10.2% | 7.2% | 82% | 16% | 8% | 7% | 3% | 23 |
| E1+E2_norm | base | norm | inv_freq | 8.8% | 8.3% | 19% | 10% | 8% | 10% | 5% | 22 |
| E1+E2+E3 | sincos | norm | inv_freq | 9.0% | 8.5% | 20% | 10% | 8% | 10% | 5% | 23 |
| E7_huber | base | raw | inv_freq | 75.7% | 58.2% | 1508% | 433% | 72% | 27% | 62% | 21 |
| E7_leaves127 | base | raw | inv_freq | 6.9% | 5.4% | 16% | 11% | 7% | 6% | 2% | 30 |
| E7_minchild10 | base | raw | inv_freq | 7.9% | 6.4% | 16% | 12% | 8% | 8% | 3% | 26 |
| E9_trees150 | base | raw | inv_freq | 9.0% | 7.8% | 17% | 12% | 8% | 10% | 4% | 9 |
| E9_trees60 | base | raw | inv_freq | 9.9% | 9.1% | 21% | 13% | 9% | 11% | 5% | 5 |
| E6_specialist | base | raw | inv_freq | 7.4% | 5.9% | 18% | 9% | 7% | 8% | 3% | 25 |
| E7_leaves255 | base | raw | inv_freq | 6.0% | 4.4% | 16% | 10% | 6% | 5% | 2% | 40 |
| E7_leaves511 | base | raw | inv_freq | 5.1% | 3.7% | 16% | 9% | 5% | 4% | 2% | 54 |
| E6+E7_l127 | base | raw | inv_freq | 6.6% | 5.1% | 19% | 8% | 7% | 7% | 2% | 32 |
| E4_density | density | raw | inv_freq | 7.8% | 6.3% | 16% | 12% | 8% | 8% | 3% | 25 |
| E4_density_l511 | density | raw | inv_freq | 5.1% | 3.6% | 17% | 9% | 5% | 4% | 2% | 63 |

_Distance MedAPE per bucket (<1km … >25km); all values on the same balanced held-out set._
