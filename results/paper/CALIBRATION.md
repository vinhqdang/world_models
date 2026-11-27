# Calibration of the paper cells (simulator-only development cohorts)

Target (authors' design): outcome-blind oracle R^2 about 50% (+- 8) on the primary group. Search: per-axis command budget sigma (command gain g = 0.98 (1 - sigma) / action limit), secant on 40 parents x 24 draws development cohorts (seed offset 500000); verification on a second disjoint cohort of 48 parents x 32 draws (offset 600000). MISALIGN: offsets 300000 / 400000. No model, no method comparison enters. New keys in results/cwk/CALIBRATION.json; the Stage 2 keys are untouched.

## Information-limited cells (INFOLIM)

| key | env | law | pi | r | iid | sigma | gain | dev oracle % | verification oracle % | verification copy % | rho | within 50 +- 8 | at search bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| info_E1o_gated_p20_r3 | E1o | gated | 0.2 | 3 | 1.0 | 0.8117 | 0.4101 | 50.0 | 45.7 | -6.0 | 0.46 | yes | no |
| info_E1o_tanh_r3 | E1o | tanh | - | 3 | 1.0 | 0.6346 | 0.7958 | 49.4 | 45.4 | -5.4 | 0.46 | yes | no |
| info_E1o_gauss_r3 | E1o | gauss | - | 3 | 1.0 | 0.7648 | 0.5122 | 50.1 | 45.3 | -3.2 | 0.43 | yes | no |
| info_E1o_gated_p5_r3 | E1o | gated | 0.05 | 3 | 1.0 | 0.8829 | 0.255 | 50.5 | 49.8 | -27.4 | 0.65 | yes | no (refined) |
| info_E1o_gated_p10_r3 | E1o | gated | 0.1 | 3 | 1.0 | 0.8578 | 0.3097 | 50.2 | 45.9 | -8.1 | 0.51 | yes | no |
| info_E1o_gated_p50_r3 | E1o | gated | 0.5 | 3 | 1.0 | 0.7256 | 0.5976 | 52.0 | 47.4 | -6.8 | 0.48 | yes | no |
| info_E1o_gated_p100_r3 | E1o | gated | 1.0 | 3 | 1.0 | 0.6529 | 0.7559 | 49.6 | 45.5 | -5.2 | 0.45 | yes | no |
| info_E1o_gated_p20_r2 | E1o | gated | 0.2 | 2 | 1.0 | 0.8 | 0.4356 | 50.2 | 51.2 | -77.0 | 1.86 | yes | no |
| info_E1o_gated_p20_r4 | E1o | gated | 0.2 | 4 | 1.0 | 0.7747 | 0.4907 | 52.0 | 48.9 | -6.9 | 0.35 | yes | no |
| info_E1o_gated_p20_r3_iid50 | E1o | gated | 0.2 | 3 | 0.5 | 0.8259 | 0.3792 | 50.2 | 44.6 | 4.9 | 0.45 | yes | no |
| info_E2o_gated_p20_r3 | E2o | gated | 0.2 | 3 | 1.0 | 0.8 | 0.56 | 50.9 | 54.6 | -7.4 | 0.48 | yes | no |
| info_E2o_tanh_r3 | E2o | tanh | - | 3 | 1.0 | 0.6229 | 1.0559 | 49.7 | 49.9 | -8.4 | 0.49 | yes | no |
| info_E2o_gauss_r3 | E2o | gauss | - | 3 | 1.0 | 0.76 | 0.672 | 50.1 | 48.9 | -5.0 | 0.49 | yes | no |
| info_E3o_gated_p20_r4 | E3o | gated | 0.2 | 4 | 1.0 | 0.8253 | 0.3567 | 50.1 | 57.6 | 18.7 | 0.52 | yes | no |
| info_E2o_gated_p5_r3 | E2o | gated | 0.05 | 3 | 1.0 | 0.9229 | 0.2159 | 50.1 | 42.7 | 21.8 | 0.54 | yes | no (refined) |
| info_E2o_gated_p10_r3 | E2o | gated | 0.1 | 3 | 1.0 | 0.8742 | 0.3522 | 51.1 | 47.1 | 4.3 | 0.52 | yes | no (refined) |
| info_E2o_gated_p50_r3 | E2o | gated | 0.5 | 3 | 1.0 | 0.7138 | 0.8014 | 50.1 | 52.1 | -9.9 | 0.51 | yes | no |
| info_E2o_gated_p100_r3 | E2o | gated | 1.0 | 3 | 1.0 | 0.6424 | 1.0013 | 50.1 | 49.5 | -8.0 | 0.49 | yes | no |
| info_E3o_tanh_r4 | E3o | tanh | - | 4 | 1.0 | 0.6929 | 0.627 | 49.1 | 51.0 | 27.8 | 0.5 | yes | no |
| info_E2o_gated_p20_r2 | E2o | gated | 0.2 | 2 | 1.0 | 0.8205 | 0.5026 | 49.7 | 47.9 | -84.3 | 2.1 | yes | no (refined) |
| info_E2o_gated_p20_r4 | E2o | gated | 0.2 | 4 | 1.0 | 0.7775 | 0.623 | 51.8 | 50.8 | 3.5 | 0.33 | yes | no |
| info_E2o_gated_p20_r3_iid50 | E2o | gated | 0.2 | 3 | 0.5 | 0.8186 | 0.5079 | 50.6 | 53.8 | 4.5 | 0.48 | yes | no |
| info_E3o_gauss_r4 | E3o | gauss | - | 4 | 1.0 | 0.8173 | 0.373 | 49.0 | 50.5 | 33.5 | 0.49 | yes | no |
| info_E3o_gated_p5_r4 | E3o | gated | 0.05 | 4 | 1.0 | 0.9199 | 0.1635 | 50.7 | 48.0 | 38.4 | 0.43 | yes | no (refined) |
| info_E3o_gated_p10_r4 | E3o | gated | 0.1 | 4 | 1.0 | 0.8959 | 0.2125 | 50.1 | 50.8 | 40.4 | 0.45 | yes | no (refined) |
| info_E3o_gated_p50_r4 | E3o | gated | 0.5 | 4 | 1.0 | 0.7603 | 0.4894 | 51.2 | 56.4 | 24.2 | 0.52 | yes | no |
| info_E3o_gated_p100_r4 | E3o | gated | 1.0 | 4 | 1.0 | 0.7093 | 0.5935 | 49.5 | 51.4 | 28.0 | 0.5 | yes | no |
| info_E3o_gated_p20_r3 | E3o | gated | 0.2 | 3 | 1.0 | 0.8543 | 0.2975 | 49.7 | 54.7 | -29.8 | 1.99 | yes | no |
| info_E3o_gated_p20_r5 | E3o | gated | 0.2 | 5 | 1.0 | 0.8 | 0.4083 | 50.3 | 57.7 | 25.4 | 0.37 | yes | no |
| info_E3o_gated_p20_r4_iid50 | E3o | gated | 0.2 | 4 | 0.5 | 0.8537 | 0.2987 | 49.9 | 56.6 | 30.6 | 0.56 | yes | no |

## Force unit (r = d + 2 cells: force scale = unit x sigma; force-only driver calibrated to oracle R^2 0.70)

| key | sigma_ref | force scale | unit |
|---|---|---|---|
| force_unit_E1o | 0.8117 | 0.7112 | 0.8762 |
| force_unit_E2o | 0.8 | 1.7653 | 2.2066 |
| force_unit_E3o | 0.8253 | 14522.1208 | 17596.172 |

## MISALIGN (Ball-in-Cup, rotate channel [+ offset], tanh law)

| key | sigma | gain | dev oracle % | verification oracle % | verification copy % | rho | within 50 +- 8 |
|---|---|---|---|---|---|---|---|
| misalign_E2o_tanh | 0.6276 | 1.0427 | 50.1 | 47.4 | -13.0 | 0.49 | yes |
| misalign_E2o_tanh_nooff | 0.6185 | 1.0682 | 49.7 | 48.9 | -95.9 | 2.02 | yes |
