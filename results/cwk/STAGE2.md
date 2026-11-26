# Stage 2: gate P-A (rank >= 2 disturbances)


## Protocol as implemented (deviations in bold)

- Cells (built from the authors' simulators; physics untouched, mechanisms composed by `cwk/sim.py`). Amplitudes were calibrated on simulator-only
  development cohorts (disjoint reset seeds, 40-48 parents x 32 draws; no model involved) to the authors' target of an outcome-blind oracle R^2 of
  about 50% on the scored group: E1 = HandReach, independent offset driver (weight 1 - nu) and gain driver (weight nu), total amplitude sigma, command gain
  g = 0.98 (1 - sigma) / 0.45: nu = 0.5 (sigma 0.758, g 0.5274), nu = 0.8 (sigma 0.802, g 0.431);
  misspecified variants nu = 0.5 with a saturating gain tanh(3x)/tanh(3) (sigma 0.742) and a deadzone gain (zero for |x| < 0.4, sigma 0.768).
  E2 = Ball-in-Cup, gain driver (sigma 0.6, g 1.12) plus an independent wind force on the ball (scale 3.0); E3 = FetchReach, gain driver on the goal command
  (sigma 0.85, g 0.306) plus an independent load on the gripper link (scale 14000.0 N).
  **Deviation 1 (E1 parametrization):** a pure-gain driver (lambda = 1) is rank one and total amplitude sigma(1 - lambda + lambda) <= 1 - g a_lim
  cannot accommodate two full-size drivers, so E1 is indexed by the gain weight nu in {0.5, 0.8} (interaction ratio rho 0.51 and 1.71, matching the authors' lambda = 0.5 and
  close to lambda = 1) instead of lambda in {0.5, 1}.
- Drivers: eps_j = tanh(driver_j), driver_j = sqrt(1 - v) field_j(t) + sqrt(v) n_t (authors' EpisodeField, temporal scale 4); v = 0.5 main cells (true lag-1 correlation
  of eps 0.464, lag-2 0.423), v = 1 for the iid cells. Episodes of 5 transitions (t = 0, 1 behaviour policy; t = 2 anchor, 3, 4 uniform actions), 16384 factual
  episodes per cell; evaluation population 512 parents x 64 draws x 3 queries (authors' protocol, state level); 64 x 32 development population (separate seeds) used only
  for the rho-interpolation reference arm.
- Mean predictor mu_hat(H, a): MLP (3 x 384, SiLU) on the last three states and the action, 2-fold cross-fitting by episode for the residuals, a model on all data for the queries.
  **Deviation 2 (compute):** 20 epochs, batch 512 (Stage 0 used 40 epochs, batch 256, width 512); kernel heads 3000 steps of batch 512; CVAE 4000 steps; 5 fit seeds throughout.
- **Deviation 3 (coordinates):** every arm models the residual of the scored coordinates only (wrist 2, cup position 2, gripper position 3), not all joint angles; a first
  version modelling all 24 joint angles under-fitted the neural head (see `cwk/cells.py`), and the choice was made on held-out factual NLL and compute (and seed-0 numbers were visible).
  FetchReach finger coordinates are fixed by the declared fixed action and are not scored.
- Kernel arms (psi(a) = (1, a_axes / a_lim) with the declared actuated axes; this is the same structural knowledge the abduction baselines use): `pck_pol` factor model rank 4, quasi-likelihood of single residuals with a learned diagonal
  nugget, transport with N = 0 and ridge tau = 1e-3 (standardised units); `pck_echo` unstructured Phi(h, a) of rank 8 fitted to consecutive-residual pairs (diagonal blocks, lag-1 products
  with a learned kappa); `pck_joint` the psi-structured factor model fitted to the same pair likelihood; `pck_rank1` rank-one polarization head (vector signed location-scale);
  `ls_signed` / `ls_signed_nug` / `ls_unsigned` per-coordinate affine signed (or absolute) scale; `flow` heteroscedastic Cholesky (full-covariance) abduction; `indist_gain` single-driver
  input-disturbance abduction with the nominal fault type (offset share, gain share, amplitude of the cell) and `indist_plain` offset-only; `indist_struct`* the exact command channels;
  `hh_binned` binned (8 k-means bins of the context) Hildreth-Houck moment estimator with Burer-Monteiro factors and the same kriging, `hh_global` one bin (the context-free closed form);
  `cvae` CF-GPS with a learned conditional VAE (z in R^4); `rho_ref`* interpolation tuned on the development population's counterfactuals; `oracle`*, `rr_exact`*, `oracle_kernel`* (Monte-Carlo
  kriging with the parent's own conditional moments, two folds over draws).
- Best baseline = highest pooled R^2 among non-starred arms that are not the merged algorithm's own variants (`pck_pol`, `pck_echo`, `pck_joint`, `hh_global`); `pck_rank1`, `hh_binned`, `cvae`
  are therefore baselines. P-A aggregation: an environment passes if, in every one of its main (iid 0.5, well-specified) cells, joint - best baseline >= 0.10 with the paired CI above 0;
  Holm over the 3 environments uses, per environment, the largest cell p-value (two-sided paired bootstrap); the gate needs >= 2 environments and the misspecified condition
  (joint >= rr_learned - 0.02 on the saturating and deadzone cells).

Physical counterfactual R^2 (%) with whole-parent x fit-seed bootstrap 95% CI (5 fit seeds). Arms marked * use privileged simulator information (exact means, exact structure, or counterfactual tuning) and are excluded from the best-baseline comparison. Differences are paired.


## Summary (primary group; R^2 %, 5 fit seeds)

| cell | iid fraction | rho | rr_learned | best baseline (R^2) | oracle kernel* | pck_pol | pck_echo | pck_joint | joint - best baseline [95% CI] | joint - pol [95% CI] | joint - best excl. binned HH [95% CI] | kappa_hat (joint; true 0.464) | overid z (mean of 5 seeds) | overid rel. discrepancy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_nu50_gain_iid50 | 0.5 | 0.51 | 69.5 | hh_binned (94.0) | 98.3 | 92.8 | 85.2 | 91.6 | -2.4 [-3.4, -1.5] | -1.2 [-2.0, -0.5] | +9.3 [+8.1, +10.5] (vs cvae) | 0.24 | 5.6 | 0.25 |
| E1_nu80_gain_iid50 | 0.5 | 1.71 | -9.2 | hh_binned (91.3) | 98.3 | 89.9 | 89.5 | 89.3 | -2.0 [-3.1, -0.9] | -0.6 [-1.6, +0.4] | +19.8 [+16.1, +24.5] (vs pck_rank1) | 0.26 | 1.1 | 0.02 |
| E2_cup_iid50 | 0.5 | 0.65 | 58.7 | hh_binned (88.4) | 93.7 | 88.6 | 86.7 | 87.7 | -0.7 [-1.5, -0.0] | -0.9 [-1.4, -0.5] | +15.7 [+12.7, +19.6] (vs ls_signed_nug) | 0.21 | 4.2 | 0.11 |
| E3_fetch_iid50 | 0.5 | 1.47 | 27.7 | hh_binned (97.5) | 100.0 | 98.1 | 97.6 | 97.7 | +0.3 [-0.1, +0.6] | -0.3 [-0.6, -0.1] | +16.7 [+15.4, +18.1] (vs pck_rank1) | 0.30 | -4.0 | 0.00 |
| E1_nu50_sat_iid50 | 0.5 | 0.83 | 50.3 | hh_binned (79.0) | 95.2 | 80.2 | 77.2 | 79.1 | +0.1 [-2.0, +1.9] | -1.0 [-2.4, +0.0] | +17.2 [+14.9, +19.9] (vs cvae) | 0.24 | 7.1 | 0.28 |
| E1_nu50_deadzone_iid50 | 0.5 | 0.34 | 78.6 | hh_binned (90.9) | 97.6 | 90.8 | 83.3 | 89.8 | -1.1 [-1.8, -0.5] | -1.1 [-1.6, -0.7] | +5.4 [+4.3, +6.5] (vs cvae) | 0.21 | 5.5 | 0.28 |
| E1_nu50_gain_iid100 | 1.0 | 0.51 | 60.2 | hh_binned (93.7) | 97.2 | 92.6 | n/a | n/a | -1.2 [-2.0, -0.3] (pol) | n/a | +19.3 [+17.2, +21.3] (vs cvae) | n/a | n/a | n/a |
| E2_cup_iid100 | 1.0 | 0.63 | 53.6 | hh_binned (87.4) | 91.7 | 88.2 | n/a | n/a | +0.8 [+0.3, +1.2] (pol) | n/a | +19.0 [+16.8, +21.7] (vs ls_signed_nug) | n/a | n/a | n/a |
| E3_fetch_iid100 | 1.0 | 1.48 | 0.5 | hh_binned (99.2) | 100.0 | 99.5 | n/a | n/a | +0.3 [+0.2, +0.4] (pol) | n/a | +22.6 [+21.4, +24.0] (vs pck_rank1) | n/a | n/a | n/a |

## E1_nu50_gain_iid50 (primary group wrist, rho = 0.51)


**wrist**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 50.0 [47.9, 52.0] |
| copy* | 31.1 [26.3, 35.6] |
| rr_exact* | 74.5 [72.5, 76.1] |
| oracle_kernel* | 98.3 [97.7, 98.8] |
| rho_ref* | 73.8 [72.1, 75.2] |
| indist_struct* | 78.9 [77.6, 80.0] |
| trained | 46.6 [44.3, 48.8] |
| rr_learned | 69.5 [67.3, 71.4] |
| pck_pol | 92.8 [91.5, 94.0] |
| pck_rank1 | 74.7 [72.8, 76.6] |
| ls_signed | 68.6 [66.2, 70.8] |
| ls_signed_nug | 71.4 [69.2, 73.2] |
| ls_unsigned | 68.6 [66.2, 70.8] |
| flow | 64.8 [62.3, 67.4] |
| indist_gain | 68.4 [66.9, 69.8] |
| indist_plain | 74.5 [73.1, 75.8] |
| hh_binned | 94.0 [93.3, 94.8] |
| hh_global | 94.2 [93.4, 95.0] |
| cvae | 82.3 [80.9, 83.7] |
| pck_echo | 85.2 [83.4, 87.3] |
| pck_joint | 91.6 [90.1, 93.1] |

pck_joint minus best baseline (hh_binned): -2.4 points, paired CI [-3.4, -1.5], p=0.002

pck_joint minus rr_learned: +22.1 [+20.3, +24.3]

pck_pol minus best baseline (hh_binned): -1.2 points, paired CI [-2.0, -0.5], p=0.002

pck_pol minus rr_learned: +23.3 [+21.5, +25.5]

pck_echo minus best baseline (hh_binned): -8.8 points, paired CI [-10.2, -7.0], p=0.002

pck_echo minus rr_learned: +15.7 [+13.8, +18.2]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (cvae, 82.3%): +9.3 points, paired CI [+8.1, +10.5]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (cvae, 82.3%): +10.5 points, paired CI [+9.4, +11.6]

Factual held-out NLL / kappa: {"nll_polar": 2.189, "kappa_polar": 0.37, "nll_polar1": 2.59, "kappa_polar1": 0.37, "nll_ls": 2.833, "kappa_ls": 0.37, "nll_flow": 2.189, "kappa_flow": 0.37, "nll_echo": 2.196, "kappa_echo": 0.229, "nll_joint": 2.197, "kappa_joint": 0.239}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.256, "lag_loss_diff_mean": 0.0236, "z": 4.1759, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2144}, {"rel_discrepancy": 0.3037, "lag_loss_diff_mean": 0.036, "z": 5.743, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2185}, {"rel_discrepancy": 0.1433, "lag_loss_diff_mean": 0.0398, "z": 8.3495, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2475}, {"rel_discrepancy": 0.2381, "lag_loss_diff_mean": 0.0331, "z": 5.6101, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2401}, {"rel_discrepancy": 0.29, "lag_loss_diff_mean": 0.0254, "z": 3.9665, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2224}]

## E1_nu80_gain_iid50 (primary group wrist, rho = 1.71)


**wrist**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 42.8 [40.6, 45.2] |
| copy* | -61.0 [-72.0, -51.1] |
| rr_exact* | 2.4 [-5.2, 9.7] |
| oracle_kernel* | 98.3 [97.9, 98.8] |
| rho_ref* | 37.5 [35.1, 40.2] |
| indist_struct* | 60.7 [58.4, 62.8] |
| trained | 36.5 [34.3, 39.0] |
| rr_learned | -9.2 [-17.2, -1.5] |
| pck_pol | 89.9 [88.1, 91.6] |
| pck_rank1 | 69.5 [64.1, 73.6] |
| ls_signed | -38.2 [-62.3, -17.1] |
| ls_signed_nug | 55.7 [28.0, 70.5] |
| ls_unsigned | -91.0 [-125.3, -48.5] |
| flow | -10.8 [-18.5, -3.6] |
| indist_gain | 55.9 [53.6, 58.2] |
| indist_plain | 22.3 [17.9, 27.2] |
| hh_binned | 91.3 [90.3, 92.3] |
| hh_global | 91.7 [90.7, 92.6] |
| cvae | 20.3 [14.7, 26.0] |
| pck_echo | 89.5 [87.9, 90.9] |
| pck_joint | 89.3 [87.6, 91.0] |

pck_joint minus best baseline (hh_binned): -2.0 points, paired CI [-3.1, -0.9], p=0.002

pck_joint minus rr_learned: +98.5 [+91.0, +106.3]

pck_pol minus best baseline (hh_binned): -1.4 points, paired CI [-2.6, -0.4], p=0.008

pck_pol minus rr_learned: +99.1 [+91.9, +106.9]

pck_echo minus best baseline (hh_binned): -1.8 points, paired CI [-2.7, -1.1], p=0.002

pck_echo minus rr_learned: +98.7 [+91.3, +106.3]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (pck_rank1, 69.5%): +19.8 points, paired CI [+16.1, +24.5]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (pck_rank1, 69.5%): +20.4 points, paired CI [+16.8, +25.1]

Factual held-out NLL / kappa: {"nll_polar": 2.069, "kappa_polar": 0.37, "nll_polar1": 2.251, "kappa_polar1": 0.37, "nll_ls": 2.555, "kappa_ls": 0.37, "nll_flow": 2.075, "kappa_flow": 0.37, "nll_echo": 2.079, "kappa_echo": 0.234, "nll_joint": 2.074, "kappa_joint": 0.263}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.0198, "lag_loss_diff_mean": -0.0052, "z": -2.0786, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2328}, {"rel_discrepancy": 0.0196, "lag_loss_diff_mean": -0.0056, "z": -2.046, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2372}, {"rel_discrepancy": 0.0346, "lag_loss_diff_mean": 0.0237, "z": 6.4548, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2446}, {"rel_discrepancy": 0.0216, "lag_loss_diff_mean": -0.0015, "z": -0.5933, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2254}, {"rel_discrepancy": 0.0248, "lag_loss_diff_mean": 0.01, "z": 3.6842, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2316}]

## E2_cup_iid50 (primary group cup_position, rho = 0.65)


**cup_position**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 44.0 [42.3, 45.5] |
| copy* | -1.2 [-6.5, 4.6] |
| rr_exact* | 63.6 [61.6, 65.6] |
| oracle_kernel* | 93.7 [93.0, 94.5] |
| rho_ref* | 65.5 [63.8, 67.0] |
| indist_struct* | 47.6 [45.6, 49.3] |
| trained | 37.8 [36.1, 39.4] |
| rr_learned | 58.7 [56.3, 60.9] |
| pck_pol | 88.6 [87.4, 89.6] |
| pck_rank1 | -3.6 [-21.5, 12.5] |
| ls_signed | 51.9 [43.2, 58.4] |
| ls_signed_nug | 72.0 [67.8, 75.1] |
| ls_unsigned | 43.2 [34.0, 51.7] |
| flow | 59.6 [57.4, 61.8] |
| indist_gain | 47.6 [45.6, 49.3] |
| indist_plain | 44.1 [42.1, 46.1] |
| hh_binned | 88.4 [87.0, 89.4] |
| hh_global | 88.0 [86.7, 89.0] |
| cvae | 66.4 [64.6, 68.2] |
| pck_echo | 86.7 [85.3, 88.0] |
| pck_joint | 87.7 [86.4, 88.7] |

pck_joint minus best baseline (hh_binned): -0.7 points, paired CI [-1.5, -0.0], p=0.051

pck_joint minus rr_learned: +29.0 [+26.4, +31.3]

pck_pol minus best baseline (hh_binned): +0.2 points, paired CI [-0.4, +0.9], p=0.535

pck_pol minus rr_learned: +29.8 [+27.2, +32.4]

pck_echo minus best baseline (hh_binned): -1.6 points, paired CI [-2.8, -0.4], p=0.020

pck_echo minus rr_learned: +28.0 [+25.5, +30.4]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (ls_signed_nug, 72.0%): +15.7 points, paired CI [+12.7, +19.6]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (ls_signed_nug, 72.0%): +16.6 points, paired CI [+13.5, +20.7]

Factual held-out NLL / kappa: {"nll_polar": 2.521, "kappa_polar": 0.37, "nll_polar1": 2.7, "kappa_polar1": 0.37, "nll_ls": 2.691, "kappa_ls": 0.37, "nll_flow": 2.524, "kappa_flow": 0.37, "nll_echo": 2.523, "kappa_echo": 0.249, "nll_joint": 2.528, "kappa_joint": 0.209}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.1653, "lag_loss_diff_mean": 0.0255, "z": 3.887, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2647}, {"rel_discrepancy": 0.1212, "lag_loss_diff_mean": 0.0197, "z": 4.0395, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2503}, {"rel_discrepancy": 0.067, "lag_loss_diff_mean": 0.012, "z": 4.0664, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2366}, {"rel_discrepancy": 0.052, "lag_loss_diff_mean": 0.0148, "z": 4.7877, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2434}, {"rel_discrepancy": 0.1533, "lag_loss_diff_mean": 0.021, "z": 4.1071, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2505}]

## E3_fetch_iid50 (primary group gripper_position, rho = 1.47)


**gripper_position**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 56.7 [54.4, 58.8] |
| copy* | 5.7 [1.1, 9.9] |
| rr_exact* | 36.3 [32.2, 40.0] |
| oracle_kernel* | 100.0 [100.0, 100.0] |
| rho_ref* | 53.7 [51.5, 56.0] |
| indist_struct* | 64.7 [62.9, 66.4] |
| trained | 49.7 [47.3, 52.0] |
| rr_learned | 27.7 [23.2, 31.7] |
| pck_pol | 98.1 [97.8, 98.3] |
| pck_rank1 | 81.1 [79.5, 82.3] |
| ls_signed | 72.4 [70.0, 74.6] |
| ls_signed_nug | 76.7 [74.6, 78.3] |
| ls_unsigned | 26.5 [21.6, 30.9] |
| flow | 29.1 [23.7, 33.4] |
| indist_gain | 64.7 [62.9, 66.4] |
| indist_plain | 44.7 [41.7, 47.3] |
| hh_binned | 97.5 [97.1, 97.8] |
| hh_global | 98.2 [98.0, 98.4] |
| cvae | 32.7 [27.4, 37.9] |
| pck_echo | 97.6 [97.3, 97.9] |
| pck_joint | 97.7 [97.3, 98.0] |

pck_joint minus best baseline (hh_binned): +0.3 points, paired CI [-0.1, +0.6], p=0.172

pck_joint minus rr_learned: +70.1 [+65.9, +74.4]

pck_pol minus best baseline (hh_binned): +0.6 points, paired CI [+0.2, +0.9], p=0.002

pck_pol minus rr_learned: +70.4 [+66.3, +74.7]

pck_echo minus best baseline (hh_binned): +0.1 points, paired CI [-0.3, +0.6], p=0.590

pck_echo minus rr_learned: +69.9 [+65.8, +74.3]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (pck_rank1, 81.1%): +16.7 points, paired CI [+15.4, +18.1]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (pck_rank1, 81.1%): +17.0 points, paired CI [+15.8, +18.4]

Factual held-out NLL / kappa: {"nll_polar": 1.416, "kappa_polar": 0.37, "nll_polar1": 1.889, "kappa_polar1": 0.37, "nll_ls": 3.828, "kappa_ls": 0.37, "nll_flow": 1.432, "kappa_flow": 0.37, "nll_echo": 1.395, "kappa_echo": 0.303, "nll_joint": 1.415, "kappa_joint": 0.299}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.0036, "lag_loss_diff_mean": -0.0095, "z": -2.8142, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.3029}, {"rel_discrepancy": 0.0044, "lag_loss_diff_mean": -0.014, "z": -3.7522, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.3068}, {"rel_discrepancy": 0.0038, "lag_loss_diff_mean": -0.0155, "z": -4.3157, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.3042}, {"rel_discrepancy": 0.0055, "lag_loss_diff_mean": -0.0178, "z": -4.1441, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.3005}, {"rel_discrepancy": 0.0035, "lag_loss_diff_mean": -0.018, "z": -5.1487, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2988}]

## E1_nu50_sat_iid50 (primary group wrist, rho = 0.83)


**wrist**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 48.5 [46.5, 50.3] |
| copy* | 8.8 [3.4, 14.3] |
| rr_exact* | 57.4 [54.8, 59.8] |
| oracle_kernel* | 95.2 [94.3, 96.2] |
| rho_ref* | 61.2 [59.3, 63.0] |
| indist_struct* | 73.1 [71.5, 74.5] |
| trained | 43.6 [41.4, 45.4] |
| rr_learned | 50.3 [47.3, 52.9] |
| pck_pol | 80.2 [78.1, 82.2] |
| pck_rank1 | 57.9 [55.1, 61.0] |
| ls_signed | 50.2 [47.2, 52.9] |
| ls_signed_nug | 53.0 [50.3, 55.6] |
| ls_unsigned | 50.2 [47.2, 52.9] |
| flow | 43.5 [39.7, 47.1] |
| indist_gain | 59.2 [57.3, 61.0] |
| indist_plain | 61.2 [59.4, 63.1] |
| hh_binned | 79.0 [77.6, 80.6] |
| hh_global | 78.9 [77.5, 80.4] |
| cvae | 61.9 [59.6, 64.3] |
| pck_echo | 77.2 [74.7, 79.6] |
| pck_joint | 79.1 [76.8, 81.4] |

pck_joint minus best baseline (hh_binned): +0.1 points, paired CI [-2.0, +1.9], p=0.859

pck_joint minus rr_learned: +28.8 [+26.1, +31.9]

pck_pol minus best baseline (hh_binned): +1.1 points, paired CI [-0.4, +2.6], p=0.164

pck_pol minus rr_learned: +29.8 [+27.4, +32.9]

pck_echo minus best baseline (hh_binned): -1.8 points, paired CI [-3.6, +0.1], p=0.062

pck_echo minus rr_learned: +26.9 [+24.1, +30.1]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (cvae, 61.9%): +17.2 points, paired CI [+14.9, +19.9]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (cvae, 61.9%): +18.3 points, paired CI [+16.2, +20.6]

Factual held-out NLL / kappa: {"nll_polar": 2.396, "kappa_polar": 0.37, "nll_polar1": 2.881, "kappa_polar1": 0.37, "nll_ls": 2.833, "kappa_ls": 0.37, "nll_flow": 2.306, "kappa_flow": 0.37, "nll_echo": 2.306, "kappa_echo": 0.248, "nll_joint": 2.403, "kappa_joint": 0.244}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.345, "lag_loss_diff_mean": 0.0292, "z": 4.4451, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2352}, {"rel_discrepancy": 0.3225, "lag_loss_diff_mean": 0.0332, "z": 5.7519, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2387}, {"rel_discrepancy": 0.2431, "lag_loss_diff_mean": 0.0534, "z": 10.1798, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2489}, {"rel_discrepancy": 0.2219, "lag_loss_diff_mean": 0.0379, "z": 7.2852, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2491}, {"rel_discrepancy": 0.2819, "lag_loss_diff_mean": 0.0499, "z": 7.6683, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2704}]

## E1_nu50_deadzone_iid50 (primary group wrist, rho = 0.34)


**wrist**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 50.3 [48.2, 52.4] |
| copy* | 43.6 [39.5, 47.6] |
| rr_exact* | 83.1 [81.6, 84.4] |
| oracle_kernel* | 97.6 [97.1, 98.1] |
| rho_ref* | 80.7 [79.3, 81.9] |
| indist_struct* | 82.7 [81.7, 83.6] |
| trained | 47.1 [44.8, 49.3] |
| rr_learned | 78.6 [76.8, 80.1] |
| pck_pol | 90.8 [90.0, 91.7] |
| pck_rank1 | 82.0 [80.6, 83.3] |
| ls_signed | 78.3 [76.5, 79.9] |
| ls_signed_nug | 79.5 [77.9, 81.0] |
| ls_unsigned | 78.3 [76.5, 79.9] |
| flow | 74.1 [71.7, 76.3] |
| indist_gain | 72.9 [71.5, 74.2] |
| indist_plain | 81.6 [80.4, 82.8] |
| hh_binned | 90.9 [90.1, 91.6] |
| hh_global | 90.5 [89.8, 91.3] |
| cvae | 84.4 [83.1, 85.5] |
| pck_echo | 83.3 [81.1, 85.2] |
| pck_joint | 89.8 [88.8, 90.8] |

pck_joint minus best baseline (hh_binned): -1.1 points, paired CI [-1.8, -0.5], p=0.002

pck_joint minus rr_learned: +11.2 [+9.6, +12.8]

pck_pol minus best baseline (hh_binned): -0.0 points, paired CI [-0.4, +0.4], p=0.836

pck_pol minus rr_learned: +12.3 [+10.7, +13.9]

pck_echo minus best baseline (hh_binned): -7.5 points, paired CI [-9.2, -6.1], p=0.002

pck_echo minus rr_learned: +4.7 [+2.5, +6.8]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_joint minus best remaining baseline (cvae, 84.4%): +5.4 points, paired CI [+4.3, +6.5]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (cvae, 84.4%): +6.5 points, paired CI [+5.5, +7.4]

Factual held-out NLL / kappa: {"nll_polar": 2.087, "kappa_polar": 0.37, "nll_polar1": 2.4, "kappa_polar1": 0.37, "nll_ls": 2.825, "kappa_ls": 0.37, "nll_flow": 1.968, "kappa_flow": 0.37, "nll_echo": 1.962, "kappa_echo": 0.221, "nll_joint": 2.099, "kappa_joint": 0.209}

Overidentification (polarization vs echo cross-kernel), per fit seed: [{"rel_discrepancy": 0.2824, "lag_loss_diff_mean": 0.0441, "z": 6.6622, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.213}, {"rel_discrepancy": 0.2391, "lag_loss_diff_mean": 0.0311, "z": 4.819, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2128}, {"rel_discrepancy": 0.169, "lag_loss_diff_mean": 0.027, "z": 4.8634, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2415}, {"rel_discrepancy": 0.3998, "lag_loss_diff_mean": 0.0545, "z": 6.9382, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.224}, {"rel_discrepancy": 0.2903, "lag_loss_diff_mean": 0.0271, "z": 4.1139, "n_pairs": 6552, "n_episodes": 1638, "kappa_echo": 0.2151}]

## E1_nu50_gain_iid100 (primary group wrist, rho = 0.51)


**wrist**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 23.0 [22.2, 23.8] |
| copy* | 22.6 [18.3, 26.5] |
| rr_exact* | 61.0 [58.7, 63.1] |
| oracle_kernel* | 97.2 [96.4, 98.1] |
| rho_ref* | 65.1 [63.6, 66.6] |
| indist_struct* | 72.4 [71.3, 73.4] |
| trained | 22.5 [21.6, 23.3] |
| rr_learned | 60.2 [58.0, 62.3] |
| pck_pol | 92.6 [91.1, 94.1] |
| pck_rank1 | 67.3 [65.2, 69.2] |
| ls_signed | 59.7 [57.4, 61.8] |
| ls_signed_nug | 61.8 [59.7, 63.8] |
| ls_unsigned | 59.7 [57.4, 61.8] |
| flow | 53.2 [50.5, 56.1] |
| indist_gain | 59.2 [58.1, 60.4] |
| indist_plain | 71.0 [69.7, 72.3] |
| hh_binned | 93.7 [92.8, 94.8] |
| hh_global | 93.6 [92.7, 94.6] |
| cvae | 73.3 [71.6, 75.3] |

pck_pol minus best baseline (hh_binned): -1.2 points, paired CI [-2.0, -0.3], p=0.008

pck_pol minus rr_learned: +32.4 [+30.4, +34.7]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (cvae, 73.3%): +19.3 points, paired CI [+17.2, +21.3]

_E1_nu80_gain_iid100: not complete_


## E2_cup_iid100 (primary group cup_position, rho = 0.63)


**cup_position**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 26.7 [25.9, 27.4] |
| copy* | 5.3 [1.0, 9.5] |
| rr_exact* | 53.8 [51.6, 55.9] |
| oracle_kernel* | 91.7 [90.8, 92.7] |
| rho_ref* | 60.7 [59.2, 62.0] |
| indist_struct* | 37.3 [35.9, 38.5] |
| trained | 25.6 [24.8, 26.3] |
| rr_learned | 53.6 [51.4, 55.6] |
| pck_pol | 88.2 [87.2, 89.3] |
| pck_rank1 | -18.0 [-37.7, 0.2] |
| ls_signed | 46.0 [37.9, 53.0] |
| ls_signed_nug | 69.2 [66.4, 71.6] |
| ls_unsigned | 34.8 [27.9, 40.8] |
| flow | 55.0 [53.0, 57.1] |
| indist_gain | 37.3 [35.9, 38.5] |
| indist_plain | 31.8 [30.3, 33.3] |
| hh_binned | 87.4 [86.4, 88.5] |
| hh_global | 87.7 [86.7, 88.7] |
| cvae | 63.3 [61.7, 64.9] |

pck_pol minus best baseline (hh_binned): +0.8 points, paired CI [+0.3, +1.2], p=0.002

pck_pol minus rr_learned: +34.6 [+32.5, +36.8]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (ls_signed_nug, 69.2%): +19.0 points, paired CI [+16.8, +21.7]

## E3_fetch_iid100 (primary group gripper_position, rho = 1.48)


**gripper_position**

| arm | R^2 % [95% CI] |
|---|---|
| oracle* | 32.7 [30.8, 34.4] |
| copy* | -7.0 [-11.5, -2.7] |
| rr_exact* | 0.6 [-3.5, 4.7] |
| oracle_kernel* | 100.0 [100.0, 100.0] |
| rho_ref* | 37.3 [35.5, 38.9] |
| indist_struct* | 71.2 [70.3, 72.2] |
| trained | 32.5 [30.7, 34.2] |
| rr_learned | 0.5 [-3.6, 4.7] |
| pck_pol | 99.5 [99.4, 99.6] |
| pck_rank1 | 76.9 [75.4, 78.2] |
| ls_signed | 10.4 [-56.6, 65.8] |
| ls_signed_nug | 68.7 [65.8, 70.9] |
| ls_unsigned | -41.0 [-96.0, 3.5] |
| flow | 4.0 [-0.6, 8.3] |
| indist_gain | 71.2 [70.3, 72.2] |
| indist_plain | 25.3 [22.4, 27.9] |
| hh_binned | 99.2 [99.1, 99.3] |
| hh_global | 99.3 [99.1, 99.5] |
| cvae | 11.8 [7.3, 16.8] |

pck_pol minus best baseline (hh_binned): +0.3 points, paired CI [+0.2, +0.4], p=0.002

pck_pol minus rr_learned: +99.0 [+94.8, +103.1]

Sensitivity (binned Hildreth-Houck BLUP removed from the baseline set): pck_pol minus best remaining baseline (pck_rank1, 76.9%): +22.6 points, paired CI [+21.4, +24.0]

_E1_nu50_sat_iid100: not complete_


_E1_nu50_deadzone_iid100: not complete_


## Reading of the results

- P-A is decided by the baseline set. The binned Hildreth-Houck BLUP (moment estimator of the same bi-affine kernel class per context bin, no network, no amortization) is within a few points of the
  neural heads in every cell (joint minus it: -2.4, -2.0, -0.7, +0.3 points on the four main cells), so no cell comes near the +10 margin. With that baseline removed, the joint estimator leads the best remaining
  baseline by +9.3 (E1 nu 0.5, CF-GPS CVAE), +19.8 (E1 nu 0.8), +15.7 (E2) and +16.7 (E3) points; the cells' rank-two structure is what the rank-one signed rules, flows, input-disturbance
  abduction and the CVAE cannot represent, but a classical estimator represents it too.
- Persistence did not help: joint is 0.3-1.2 points below polarization-only in the main cells, echo-only is 0.4-7.6 points below it, and the estimated persistence kappa_hat (0.2-0.3) is about half of the true
  lag-1 correlation of the driver (0.464). A plausible cause (not tested here) is that the mean predictor sees the previous states, which reveal the previous disturbance, and so absorbs part of the persistent
  component before the residuals are formed; lagged residual products then identify an attenuated kappa. The overidentification statistic between the polarization-implied and echo-implied cross-kernels (z of the held-out lag-product loss difference,
  and the relative discrepancy of the two cross-kernels) is large in five of the six iid 0.5 cells (|z| 4.0-7.1; relative discrepancy 0.11-0.28 where z is positive; E1 nu 0.8 is quiet at z 1.1) although the polarization-implied kernel is the accurate one
  (it ties the binned moment estimator and sits within 1-6 points of the oracle kernel): as implemented the statistic reacts to the estimation problem of the echo route (attenuated kappa_hat, an unstructured network), not to
  misspecification of the bi-affine class, and is not calibrated (Stage 3 was not attempted).
- Misspecified psi (saturating gain, deadzone): the joint estimator stays far above residual reuse (+28.8 and +11.2 points) and ties the binned moment estimator, but loses 16 and 8 points to the exact-structure oracle kernel.
- iid_fraction = 1.0 cells (polarization only): polarization-only minus the best baseline (binned HH) is +0.8 (E2), +0.3 (E3) and -1.2 (E1 nu 0.5), i.e. again a tie with the classical estimator, and +19 to +23 points over the best baseline that is not a moment estimator of the same class.


# P-A verdict

- E1: effect and CI criterion NOT met; Holm-adjusted p = 0.0059 (reject); E1_nu50_gain_iid50: joint - hh_binned = -2.4 pts, CI [-3.4, -1.5]; E1_nu80_gain_iid50: joint - hh_binned = -2.0 pts, CI [-3.1, -0.9]
- E2: effect and CI criterion NOT met; Holm-adjusted p = 0.1016 (not significant); E2_cup_iid50: joint - hh_binned = -0.7 pts, CI [-1.5, -0.0]
- E3: effect and CI criterion NOT met; Holm-adjusted p = 0.1719 (not significant); E3_fetch_iid50: joint - hh_binned = +0.3 pts, CI [-0.1, +0.6]
- misspecified E1_nu50_sat_iid50: joint - rr_learned = +28.8 pts [+26.1, +31.9] -> ok; joint - rr_exact = +21.7 pts
- misspecified E1_nu50_deadzone_iid50: joint - rr_learned = +11.2 pts [+9.6, +12.8] -> ok; joint - rr_exact = +6.7 pts

**Environments passing (strict: all main cells, Holm alpha 0.05): [] (0/3). Misspecified-psi condition: met. GATE P-A: FAIL.**

Sensitivity (NOT the protocol's verdict): with the binned Hildreth-Houck BLUP removed from the baseline set the same rule gives environments passing = ['E2', 'E3'] (E1 fails through its nu = 0.5 cell, +9.3 [+8.1, +10.5]); the gate would then be PASS. The protocol lists the binned Hildreth-Houck BLUP as a baseline, so the official verdict above stands.