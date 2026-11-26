# Stage 1: honesty check on rank-one cells


## Setup

- Analytic dial (`cwk/analytic.py`): x_{t+1} = 0.5 x_t + m(c_t) { g a + sigma eps d (1 - lambda + lambda a) } (the dial of App. D eq. 39 per coordinate), eps = tanh of a Gaussian driver, d = (+1, -1),
  actions uniform on [-1, 1]^d, an exogenous context c_t modulating the scale (m = 1 + 0.3 tanh c), sigma 0.609, g 0.383; iid driver (fresh noise only); 6000 factual episodes x 5 transitions, evaluation
  population 256 x 64 x 3 queries; d = 1 (scalar, the case in which rank-one PCK and the signed location-scale rule coincide) and d = 2. Only 6 of the 8 planned analytic cells were run
  (d = 2: lambda 0, 0.5, 1; d = 1: lambda 0, 0.5, 1) because of compute contention.
- HandReach dial cells: the authors' hand (lambda = 0, 512 parents x 128 draws), hand_affine50 and hand_affine100 (512 x 64), 8192 factual episodes x 5 transitions, state level, iid (v = 1).
- Pipeline as in Stage 2 (see STAGE2.md): MLP mean predictor with 2-fold cross-fitting (HandReach: width 384, 25 epochs, batch 512; analytic: width 256, 20 epochs), kernel heads on the scored coordinates
  (wrist; analytic: all outcome coordinates), 5 fit seeds, whole-parent x seed paired bootstrap. Baselines as listed in STAGE2.md; `indist_gain` uses the exact fault type of the cell (lambda, sigma).
- Rows whose `pck_rank1` is `n/a` were produced before a fix of the rank-one head (random restarts, see `cwk/kernels.py`) and the arm was removed; the d = 1 cells do not need it (R = 1 is the signed scale).


## Summary: polarization-only PCK against the signed location-scale rules

Differences in R^2 points (paired bootstrap 95% CI). `ls_signed`: per-coordinate signed affine scale with ridge 1e-3 (the reviewer's rule, applied coordinatewise); `ls_signed_nug`: same with the fitted nugget as the ridge; `pck_rank1`: the vector-valued rank-one rule (one signed scale per action, shared by all coordinates), i.e. PCK with R = 1; `pck_pol`: R = 4. Scalar cells (d1) are the case in which the signed location-scale rule and rank-one PCK coincide.

| cell | rho | pck_pol | pck_rank1 | ls_signed | ls_signed_nug | ls_unsigned | rr_learned | pol - ls_signed | pol - ls_signed_nug | pol - rank1 | pol - rr_learned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dial_d2_l0 | 0.00 | 99.5 | n/a | 99.5 | 99.4 | 99.5 | 99.7 | +0.0 [-0.1, +0.1] | +0.2 [+0.1, +0.3] | n/a | -0.2 [-0.3, -0.1] |
| dial_d2_l50 | 0.52 | 99.3 | 99.3 | 93.9 | 96.7 | 95.0 | 81.0 | +5.4 [+3.1, +8.1] | +2.6 [+1.7, +3.4] | -0.0 [-0.2, +0.1] | +18.2 [+16.8, +19.7] |
| dial_d2_l100 | 2.09 | 99.6 | 99.5 | 97.2 | 97.5 | 20.1 | 17.6 | +2.4 [+1.5, +3.6] | +2.1 [+1.5, +2.7] | +0.1 [+0.0, +0.2] | +82.0 [+75.9, +88.1] |
| dial_d1_l0 | 0.00 | 99.6 | n/a | 99.7 | 99.5 | 99.7 | 99.8 | -0.1 [-0.2, +0.0] | +0.1 [+0.0, +0.2] | n/a | -0.2 [-0.3, -0.1] |
| dial_d1_l50 | 0.54 | 97.0 | n/a | 96.3 | 97.7 | 96.4 | 79.8 | +0.7 [+0.2, +1.5] | -0.7 [-1.7, -0.1] | n/a | +17.1 [+14.8, +19.2] |
| dial_d1_l100 | 1.91 | 97.9 | n/a | 97.7 | 98.0 | 21.1 | 24.2 | +0.1 [-0.1, +0.6] | -0.1 [-1.9, +0.6] | n/a | +73.7 [+64.8, +83.8] |
| hand_l0 | 0.03 | 96.4 | n/a | 97.5 | 97.9 | 97.6 | 97.6 | -1.1 [-1.9, +0.1] | -1.5 [-2.0, -1.1] | n/a | -1.2 [-1.7, -0.7] |
| hand_affine50 | 0.46 | 96.1 | 96.9 | 84.8 | 93.4 | 86.9 | 76.8 | +11.3 [+7.1, +16.2] | +2.7 [+1.9, +3.6] | -0.8 [-1.5, -0.2] | +19.3 [+18.2, +20.6] |
| hand_affine100 | 2.06 | 96.6 | 97.4 | 80.0 | 88.1 | -10.3 | 2.7 | +16.6 [+8.5, +29.2] | +8.5 [+3.4, +17.8] | -0.8 [-1.1, -0.5] | +94.0 [+88.8, +99.1] |
R^2 (%) of the counterfactual on the scored group, 5 fit seeds, whole-parent x seed bootstrap 95% CI. `pck_pol` = polarization-only PCK. Differences are paired (same fit seeds, same bootstrap resamples).


## Analytic dial (App. D eq. 39 analogue, d = 1 or 2 coordinates)


**dial_d2_l0 / all** (rho = 0.00)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 40.7 | [38.5, 42.7] |
| copy | 59.6 | [56.4, 62.4] |
| rr_exact | 100.0 | [100.0, 100.0] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 40.4 | [38.3, 42.4] |
| rr_learned | 99.7 | [99.6, 99.8] |
| pck_pol | 99.5 | [99.4, 99.6] |
| ls_signed | 99.5 | [99.4, 99.6] |
| ls_signed_nug | 99.4 | [99.2, 99.5] |
| ls_unsigned | 99.5 | [99.4, 99.6] |
| flow | 99.7 | [99.6, 99.8] |
| indist_gain | 92.7 | [91.7, 93.5] |
| indist_plain | 92.7 | [91.7, 93.5] |
| hh_global | 99.7 | [99.6, 99.7] |

PCK-pol minus ls_signed: +0.0 points, paired 95% CI [-0.1, +0.1]

PCK-pol minus ls_signed_nug: +0.2 points, paired 95% CI [+0.1, +0.3]

PCK-pol minus rr_learned: -0.2 points, paired 95% CI [-0.3, -0.1]

PCK-pol minus ls_unsigned: +0.0 points, paired 95% CI [-0.1, +0.1]


_dial_d2_l25: 0/5 seeds complete_


**dial_d2_l50 / all** (rho = 0.52)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 64.2 | [62.3, 66.1] |
| copy | 7.4 | [0.1, 14.8] |
| rr_exact | 81.3 | [79.9, 82.7] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 64.0 | [62.1, 65.8] |
| rr_learned | 81.0 | [79.5, 82.5] |
| pck_pol | 99.3 | [98.8, 99.6] |
| pck_rank1 | 99.3 | [98.9, 99.6] |
| ls_signed | 93.9 | [91.1, 96.3] |
| ls_signed_nug | 96.7 | [95.8, 97.6] |
| ls_unsigned | 95.0 | [92.2, 96.9] |
| flow | 95.9 | [94.1, 97.1] |
| indist_gain | 91.7 | [90.5, 93.0] |
| indist_plain | 85.8 | [84.8, 86.9] |
| hh_global | 99.3 | [98.9, 99.5] |

PCK-pol minus pck_rank1: -0.0 points, paired 95% CI [-0.2, +0.1]

PCK-pol minus ls_signed: +5.4 points, paired 95% CI [+3.1, +8.1]

PCK-pol minus ls_signed_nug: +2.6 points, paired 95% CI [+1.7, +3.4]

PCK-pol minus rr_learned: +18.2 points, paired 95% CI [+16.8, +19.7]

PCK-pol minus ls_unsigned: +4.3 points, paired 95% CI [+2.6, +7.1]


_dial_d2_l75: 0/5 seeds complete_


**dial_d2_l100 / all** (rho = 2.09)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 60.9 | [59.1, 62.9] |
| copy | -64.1 | [-76.3, -51.5] |
| rr_exact | 18.1 | [11.9, 24.3] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 60.5 | [58.7, 62.5] |
| rr_learned | 17.6 | [11.5, 23.7] |
| pck_pol | 99.6 | [99.4, 99.7] |
| pck_rank1 | 99.5 | [99.3, 99.6] |
| ls_signed | 97.2 | [96.0, 98.2] |
| ls_signed_nug | 97.5 | [96.8, 98.1] |
| ls_unsigned | 20.1 | [13.4, 26.4] |
| flow | 15.7 | [8.4, 23.3] |
| indist_gain | 91.5 | [90.2, 92.4] |
| indist_plain | 48.2 | [44.3, 51.5] |
| hh_global | 99.5 | [99.3, 99.6] |

PCK-pol minus pck_rank1: +0.1 points, paired 95% CI [+0.0, +0.2]

PCK-pol minus ls_signed: +2.4 points, paired 95% CI [+1.5, +3.6]

PCK-pol minus ls_signed_nug: +2.1 points, paired 95% CI [+1.5, +2.7]

PCK-pol minus rr_learned: +82.0 points, paired 95% CI [+75.9, +88.1]

PCK-pol minus ls_unsigned: +79.5 points, paired 95% CI [+73.1, +86.1]


**dial_d1_l0 / all** (rho = 0.00)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 40.9 | [38.3, 43.7] |
| copy | 59.8 | [55.2, 64.3] |
| rr_exact | 100.0 | [100.0, 100.0] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 40.9 | [38.2, 43.6] |
| rr_learned | 99.8 | [99.7, 99.9] |
| pck_pol | 99.6 | [99.5, 99.7] |
| ls_signed | 99.7 | [99.5, 99.8] |
| ls_signed_nug | 99.5 | [99.4, 99.6] |
| ls_unsigned | 99.7 | [99.5, 99.8] |
| flow | 99.8 | [99.7, 99.9] |
| indist_gain | 85.3 | [83.9, 86.3] |
| indist_plain | 85.3 | [83.9, 86.3] |
| hh_global | 99.8 | [99.7, 99.9] |

PCK-pol minus ls_signed: -0.1 points, paired 95% CI [-0.2, +0.0]

PCK-pol minus ls_signed_nug: +0.1 points, paired 95% CI [+0.0, +0.2]

PCK-pol minus rr_learned: -0.2 points, paired 95% CI [-0.3, -0.1]

PCK-pol minus ls_unsigned: -0.1 points, paired 95% CI [-0.2, +0.0]


**dial_d1_l50 / all** (rho = 0.54)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 63.1 | [60.4, 65.4] |
| copy | 2.5 | [-7.4, 13.7] |
| rr_exact | 80.0 | [77.9, 82.3] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 63.0 | [60.2, 65.2] |
| rr_learned | 79.8 | [77.7, 82.1] |
| pck_pol | 97.0 | [94.8, 98.4] |
| ls_signed | 96.3 | [93.5, 98.0] |
| ls_signed_nug | 97.7 | [96.3, 98.6] |
| ls_unsigned | 96.4 | [93.6, 98.1] |
| flow | 97.2 | [94.9, 98.5] |
| indist_gain | 83.7 | [81.4, 85.6] |
| indist_plain | 82.9 | [81.2, 84.4] |
| hh_global | 97.8 | [96.4, 98.7] |

PCK-pol minus ls_signed: +0.7 points, paired 95% CI [+0.2, +1.5]

PCK-pol minus ls_signed_nug: -0.7 points, paired 95% CI [-1.7, -0.1]

PCK-pol minus rr_learned: +17.1 points, paired 95% CI [+14.8, +19.2]

PCK-pol minus ls_unsigned: +0.6 points, paired 95% CI [+0.1, +1.4]


**dial_d1_l100 / all** (rho = 1.91)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 60.3 | [57.2, 63.5] |
| copy | -47.5 | [-65.7, -30.7] |
| rr_exact | 24.3 | [14.2, 33.2] |
| oracle_kernel | 100.0 | [100.0, 100.0] |
| trained | 60.1 | [57.0, 63.3] |
| rr_learned | 24.2 | [14.2, 33.1] |
| pck_pol | 97.9 | [95.4, 99.1] |
| ls_signed | 97.7 | [94.8, 99.0] |
| ls_signed_nug | 98.0 | [96.9, 98.8] |
| ls_unsigned | 21.1 | [9.7, 30.6] |
| flow | 21.1 | [10.4, 31.1] |
| indist_gain | 81.5 | [79.1, 83.7] |
| indist_plain | 39.9 | [32.4, 46.6] |
| hh_global | 97.3 | [96.1, 98.2] |

PCK-pol minus ls_signed: +0.1 points, paired 95% CI [-0.1, +0.6]

PCK-pol minus ls_signed_nug: -0.1 points, paired 95% CI [-1.9, +0.6]

PCK-pol minus rr_learned: +73.7 points, paired 95% CI [+64.8, +83.8]

PCK-pol minus ls_unsigned: +76.7 points, paired 95% CI [+67.1, +86.4]


## HandReach dial cells (authors configurations, state level)


**hand_l0 / wrist** (rho = 0.03)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 51.9 | [50.8, 52.8] |
| copy | 5.4 | [0.3, 11.1] |
| rr_exact | 98.5 | [98.2, 98.7] |
| oracle_kernel | 99.9 | [99.9, 99.9] |
| trained | 51.2 | [50.1, 52.1] |
| rr_learned | 97.6 | [97.3, 97.9] |
| pck_pol | 96.4 | [95.9, 96.9] |
| ls_signed | 97.5 | [96.4, 98.1] |
| ls_signed_nug | 97.9 | [97.7, 98.2] |
| ls_unsigned | 97.6 | [96.8, 98.1] |
| flow | 98.9 | [98.8, 99.0] |
| indist_gain | 92.6 | [92.3, 92.9] |
| indist_plain | 92.6 | [92.3, 92.9] |
| hh_global | 98.1 | [97.9, 98.4] |

PCK-pol minus ls_signed: -1.1 points, paired 95% CI [-1.9, +0.1]

PCK-pol minus ls_signed_nug: -1.5 points, paired 95% CI [-2.0, -1.1]

PCK-pol minus rr_learned: -1.2 points, paired 95% CI [-1.7, -0.7]

PCK-pol minus ls_unsigned: -1.2 points, paired 95% CI [-1.9, -0.4]


**hand_affine50 / wrist** (rho = 0.46)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 52.2 | [50.8, 53.4] |
| copy | -18.3 | [-24.2, -12.2] |
| rr_exact | 78.1 | [77.0, 79.2] |
| oracle_kernel | 99.9 | [99.9, 99.9] |
| trained | 51.1 | [49.8, 52.4] |
| rr_learned | 76.8 | [75.7, 77.9] |
| pck_pol | 96.1 | [95.3, 96.9] |
| pck_rank1 | 96.9 | [95.9, 97.6] |
| ls_signed | 84.8 | [79.6, 89.3] |
| ls_signed_nug | 93.4 | [92.4, 94.2] |
| ls_unsigned | 86.9 | [83.0, 90.5] |
| flow | 92.7 | [91.6, 93.7] |
| indist_gain | 89.1 | [88.1, 90.0] |
| indist_plain | 81.8 | [80.9, 82.6] |
| hh_global | 96.6 | [96.0, 97.1] |

PCK-pol minus pck_rank1: -0.8 points, paired 95% CI [-1.5, -0.2]

PCK-pol minus ls_signed: +11.3 points, paired 95% CI [+7.1, +16.2]

PCK-pol minus ls_signed_nug: +2.7 points, paired 95% CI [+1.9, +3.6]

PCK-pol minus rr_learned: +19.3 points, paired 95% CI [+18.2, +20.6]

PCK-pol minus ls_unsigned: +9.2 points, paired 95% CI [+5.7, +13.2]


**hand_affine100 / wrist** (rho = 2.06)

| arm | R^2 % | 95% CI |
|---|---|---|
| oracle | 53.5 | [52.6, 54.2] |
| copy | -95.1 | [-105.3, -84.7] |
| rr_exact | 4.0 | [-1.1, 9.0] |
| oracle_kernel | 99.9 | [99.9, 99.9] |
| trained | 52.5 | [51.7, 53.2] |
| rr_learned | 2.7 | [-2.5, 7.6] |
| pck_pol | 96.6 | [96.2, 97.0] |
| pck_rank1 | 97.4 | [97.1, 97.7] |
| ls_signed | 80.0 | [67.3, 88.2] |
| ls_signed_nug | 88.1 | [78.6, 93.2] |
| ls_unsigned | -10.3 | [-17.3, -4.5] |
| flow | 3.2 | [-2.4, 8.7] |
| indist_gain | 88.3 | [87.5, 89.2] |
| indist_plain | 33.3 | [29.5, 36.9] |
| hh_global | 96.8 | [96.5, 97.2] |

PCK-pol minus pck_rank1: -0.8 points, paired 95% CI [-1.1, -0.5]

PCK-pol minus ls_signed: +16.6 points, paired 95% CI [+8.5, +29.2]

PCK-pol minus ls_signed_nug: +8.5 points, paired 95% CI [+3.4, +17.8]

PCK-pol minus rr_learned: +94.0 points, paired 95% CI [+88.8, +99.1]

PCK-pol minus ls_unsigned: +107.0 points, paired 95% CI [+101.4, +113.8]
