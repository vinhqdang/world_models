| statistic | picked model | success of pick | fall of pick | success regret vs best | fall excess vs safest |
|---|---|---|---|---|---|
| ES1 (min) | es_s1 | 0.449 | 0.316 | 0.280 | 0.185 |
| MSE1 (min) | det_s2 | 0.619 | 0.370 | 0.109 | 0.238 |
| NLL1 (min) | es_s1 | 0.449 | 0.316 | 0.280 | 0.185 |
| ES_H (min) | gauss_s2 | 0.573 | 0.176 | 0.156 | 0.044 |
| SE_H (min) | gauss_s2 | 0.573 | 0.176 | 0.156 | 0.044 |
| ES1_edge (min) | es_s1 | 0.449 | 0.316 | 0.280 | 0.185 |
| Brier_hazard (min) | es_s0 | 0.507 | 0.257 | 0.221 | 0.126 |
| pool_pick_gain (min) | det_s2 | 0.619 | 0.370 | 0.109 | 0.238 |
| pool_pick_opt (min) | es_k500_s11 | 0.172 | 0.131 | 0.556 | 0.000 |
| pool_pick_fall_gain (min) | es_k1500_s10 | 0.423 | 0.220 | 0.305 | 0.088 |
| OW_ES1 (min) | es_s0 | 0.507 | 0.257 | 0.221 | 0.126 |
| OW_Brier_hazard (min) | det_s2 | 0.619 | 0.370 | 0.109 | 0.238 |
| pool_rmse (min) | gauss_k500_s11 | 0.468 | 0.161 | 0.260 | 0.030 |
| pool_tau_cell (max) | det_s2 | 0.619 | 0.370 | 0.109 | 0.238 |
| im_succ (max) | det_k2000_s10 | 0.612 | 0.350 | 0.116 | 0.219 |
| random model (expected) | - | 0.520 | 0.247 | 0.208 | 0.116 |
| oracle best-success model | det_low_s11 | 0.728 | 0.240 | 0 | - |
| oracle safest model | es_k500_s11 | 0.172 | 0.131 | - | 0 |

Kind-only oracle (score = mean ground truth of the model family): tau with success +0.75, tau with -fall +0.63.
Family means: det: succ 0.651 fall 0.320, es: succ 0.414 fall 0.237, gauss: succ 0.518 fall 0.186
det: Spearman(steps, success) -0.42, Spearman(steps, fall) +0.68 (n=9)
es: Spearman(steps, success) +0.69, Spearman(steps, fall) +0.77 (n=11)
gauss: Spearman(steps, success) +0.60, Spearman(steps, fall) +0.58 (n=9)