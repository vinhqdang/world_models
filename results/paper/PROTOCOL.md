# Protocol of the analysis-paper experiments (written before any R1-R6 run)

Plan: `paper_cf/PLAN.md` (sections 2 and 3). Code: `cwk/` (sim.py, worlds.py, cells.py, eb.py, pipeline.py, oracle.py, runner.py),
`experiments/paper_*.py`. Phase 1 (code, R0 unit tests, calibration, this protocol) is complete; no result of R1-R6 exists when this
file is written. The sealed cell is drawn after this file is committed (section 9). Nothing below is changed after results are seen;
every later deviation is recorded in a section "Deviations" appended to the corresponding R*.md.

## 1. Setting

State-level same-disturbance counterfactual transport on the authors' simulators (physics untouched, upstream commit 5c84bba, clone
`/home/user/cwm_upstream`, unmodified), protocol of Stage 2 (`results/cwk/STAGE2.md`): episodes of 5 transitions (t = 0, 1 behaviour policy;
t = 2 anchor; t = 3, 4 uniform actions), 16384 factual episodes per cell; evaluation population 512 parents x 64 draws x 3 query actions
(plus the executed action), counterfactual R^2 on the primary coordinate group (wrist d = 2, cup position d = 2, gripper position d = 3);
development population 64 x 32 (disjoint reset seeds, used only by privileged reference arms). Mean predictor mu_hat(H, a): MLP 3 x 384
(SiLU) on the last three states and the action, 20 epochs, batch 512, 2-fold cross-fitting by episode (residuals), a model on all data for queries.
Kernel / EB arms work on the residual of the scored coordinates in standardised units; psi(a) = (1, a_axes / a_lim) (declared actuated axes).

## 2. Cells

### 2.1 Reused Stage 2 cells (cached data in results/cwk/data; amplitudes from Stage 2 calibration, unchanged)

R1 and R6 use the nine cells with cached data: E1_nu50_gain_iid50, E1_nu80_gain_iid50, E2_cup_iid50, E3_fetch_iid50, E1_nu50_sat_iid50,
E1_nu50_deadzone_iid50 (iid fraction 0.5) and E1_nu50_gain_iid100, E2_cup_iid100, E3_fetch_iid100 (iid fraction 1). Amplitudes (calibrated in Stage 2):

```json
{"e1_nu50_gain": {"sigma": 0.758, "gain": 0.5274}, "e1_nu80_gain": {"sigma": 0.802, "gain": 0.431}, "e1_nu50_sat": {"sigma": 0.742, "gain": 0.5614}, "e1_nu50_deadzone": {"sigma": 0.768, "gain": 0.5056}, "e2": {"sigma": 0.6, "gain": 1.12, "wind": 3.0}, "e3": {"sigma": 0.85, "gain": 0.3062500000000001, "load": 14000.0}}
```

### 2.2 Information-limited cells INFOLIM(env, law, pi, r) (new)

Environments E1o = HandReach (d = 2, wrist; upstream config hand_affine100 with the declared axes 0, 1), E2o = Ball-in-Cup (d = 2, cup position),
E3o = FetchReach (d = 3, gripper position; goal-position command). Drivers (independent, all with the same law): r = d gains, one per
actuator (command perturbation s eps x_j on actuator j only: channel kind `gain`, key `axis` of cwk/sim.py); r = d + 1 adds a common offset
(direction = the authors' alternating +-1 on the declared axes); r = d + 2 adds a force on a body (hand: palm, vector (9, 0, 30); cup: ball, +x;
fetch: gripper link, +x). Per-axis command budget sigma: gain channels sigma (1 - 1/2), offset sigma / 2 (r = d: gains sigma); command gain
g = 0.98 (1 - sigma) / a_lim, so the disturbed command never leaves the native range (checked by R0). Force scale = force_unit x sigma.
Driver laws (cwk/sim.py `law_eps`): `tanh` eps = tanh(driver) (default); `gated(pi)` eps = gate x (2 Phi(driver) - 1) with gate ~ Bern(pi)
from a separate deterministic stream per (seed, parent, draw, driver), shared by all counterfactual branches of a draw (R0 verifies:
prefix stability, rate, independence, replay of every branch with the stored eps); `gauss_clip` eps = clip(driver / 3, -1, 1). For
iid fraction 1 the driver is N(0, 1), so the gated law is Bern(pi) x U(-1, 1). The driver of the persistent cells (Piid50) mixes the
authors' episode field with innovations (iid fraction 0.5), the gate being independent across transitions.

Cells (R2a; names `<env>_<cell>`):

* P: gated pi = 0.2, r = d + 1; T: tanh, r = d + 1; G: gauss_clip, r = d + 1;
* pi5, pi10, pi50, pi100: gated pi = 0.05, 0.1, 0.5, 1 at r = d + 1; rd, rd2: gated pi = 0.2 at r = d and r = d + 2;
* Piid50: P at iid fraction 0.5; Pdata1, Pdata2: two further data realisations of P (reset and driver seeds differ by 1e6, 2e6);
* SEALED: one cell drawn after the protocol commit (section 9).

| cell | law | iid fraction | drivers r (channels / forces) | action gain | seed offset |
|---|---|---|---|---|---|
| E1o_P | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4058; gain[axis 1] 0.4058; offset 0.4058 | 0.4101 | 8000000 |
| E1o_T | tanh | 1.0 | 3: gain[axis 0] 0.3173; gain[axis 1] 0.3173; offset 0.3173 | 0.7958 | 8000000 |
| E1o_G | gauss_clip c=3.0 | 1.0 | 3: gain[axis 0] 0.3824; gain[axis 1] 0.3824; offset 0.3824 | 0.5122 | 8000000 |
| E1o_pi5 | gated pi=0.05 | 1.0 | 3: gain[axis 0] 0.4415; gain[axis 1] 0.4415; offset 0.4415 | 0.2550 | 8000000 |
| E1o_pi10 | gated pi=0.1 | 1.0 | 3: gain[axis 0] 0.4289; gain[axis 1] 0.4289; offset 0.4289 | 0.3097 | 8000000 |
| E1o_pi50 | gated pi=0.5 | 1.0 | 3: gain[axis 0] 0.3628; gain[axis 1] 0.3628; offset 0.3628 | 0.5976 | 8000000 |
| E1o_pi100 | gated pi=1.0 | 1.0 | 3: gain[axis 0] 0.3265; gain[axis 1] 0.3265; offset 0.3265 | 0.7559 | 8000000 |
| E1o_rd | gated pi=0.2 | 1.0 | 2: gain[axis 0] 0.8000; gain[axis 1] 0.8000 | 0.4356 | 8000000 |
| E1o_rd2 | gated pi=0.2 | 1.0 | 4: gain[axis 0] 0.3874; gain[axis 1] 0.3874; offset 0.3874; force(robot0:palm) 0.6788 | 0.4907 | 8000000 |
| E1o_Piid50 | gated pi=0.2 | 0.5 | 3: gain[axis 0] 0.4129; gain[axis 1] 0.4129; offset 0.4129 | 0.3792 | 8000000 |
| E1o_Pdata1 | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4058; gain[axis 1] 0.4058; offset 0.4058 | 0.4101 | 9000000 |
| E1o_Pdata2 | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4058; gain[axis 1] 0.4058; offset 0.4058 | 0.4101 | 10000000 |
| E2o_P | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4000; gain[axis 1] 0.4000; offset 0.4000 | 0.5600 | 8100000 |
| E2o_T | tanh | 1.0 | 3: gain[axis 0] 0.3115; gain[axis 1] 0.3115; offset 0.3115 | 1.0559 | 8100000 |
| E2o_G | gauss_clip c=3.0 | 1.0 | 3: gain[axis 0] 0.3800; gain[axis 1] 0.3800; offset 0.3800 | 0.6720 | 8100000 |
| E2o_pi5 | gated pi=0.05 | 1.0 | 3: gain[axis 0] 0.4615; gain[axis 1] 0.4615; offset 0.4615 | 0.2159 | 8100000 |
| E2o_pi10 | gated pi=0.1 | 1.0 | 3: gain[axis 0] 0.4371; gain[axis 1] 0.4371; offset 0.4371 | 0.3522 | 8100000 |
| E2o_pi50 | gated pi=0.5 | 1.0 | 3: gain[axis 0] 0.3569; gain[axis 1] 0.3569; offset 0.3569 | 0.8014 | 8100000 |
| E2o_pi100 | gated pi=1.0 | 1.0 | 3: gain[axis 0] 0.3212; gain[axis 1] 0.3212; offset 0.3212 | 1.0013 | 8100000 |
| E2o_rd | gated pi=0.2 | 1.0 | 2: gain[axis 0] 0.8205; gain[axis 1] 0.8205 | 0.5026 | 8100000 |
| E2o_rd2 | gated pi=0.2 | 1.0 | 4: gain[axis 0] 0.3887; gain[axis 1] 0.3887; offset 0.3887; force(ball) 1.716 | 0.6230 | 8100000 |
| E2o_Piid50 | gated pi=0.2 | 0.5 | 3: gain[axis 0] 0.4093; gain[axis 1] 0.4093; offset 0.4093 | 0.5079 | 8100000 |
| E2o_Pdata1 | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4000; gain[axis 1] 0.4000; offset 0.4000 | 0.5600 | 9100000 |
| E2o_Pdata2 | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.4000; gain[axis 1] 0.4000; offset 0.4000 | 0.5600 | 10100000 |
| E3o_P | gated pi=0.2 | 1.0 | 4: gain[axis 0] 0.4127; gain[axis 1] 0.4127; gain[axis 2] 0.4127; offset 0.4127 | 0.3567 | 8200000 |
| E3o_T | tanh | 1.0 | 4: gain[axis 0] 0.3464; gain[axis 1] 0.3464; gain[axis 2] 0.3464; offset 0.3464 | 0.6270 | 8200000 |
| E3o_G | gauss_clip c=3.0 | 1.0 | 4: gain[axis 0] 0.4087; gain[axis 1] 0.4087; gain[axis 2] 0.4087; offset 0.4087 | 0.3730 | 8200000 |
| E3o_pi5 | gated pi=0.05 | 1.0 | 4: gain[axis 0] 0.4600; gain[axis 1] 0.4600; gain[axis 2] 0.4600; offset 0.4600 | 0.1635 | 8200000 |
| E3o_pi10 | gated pi=0.1 | 1.0 | 4: gain[axis 0] 0.4480; gain[axis 1] 0.4480; gain[axis 2] 0.4480; offset 0.4480 | 0.2125 | 8200000 |
| E3o_pi50 | gated pi=0.5 | 1.0 | 4: gain[axis 0] 0.3801; gain[axis 1] 0.3801; gain[axis 2] 0.3801; offset 0.3801 | 0.4894 | 8200000 |
| E3o_pi100 | gated pi=1.0 | 1.0 | 4: gain[axis 0] 0.3547; gain[axis 1] 0.3547; gain[axis 2] 0.3547; offset 0.3547 | 0.5935 | 8200000 |
| E3o_rd | gated pi=0.2 | 1.0 | 3: gain[axis 0] 0.8543; gain[axis 1] 0.8543; gain[axis 2] 0.8543 | 0.2975 | 8200000 |
| E3o_rd2 | gated pi=0.2 | 1.0 | 5: gain[axis 0] 0.4000; gain[axis 1] 0.4000; gain[axis 2] 0.4000; offset 0.4000; force(robot0:gripper_link) 1.408e+04 | 0.4083 | 8200000 |
| E3o_Piid50 | gated pi=0.2 | 0.5 | 4: gain[axis 0] 0.4269; gain[axis 1] 0.4269; gain[axis 2] 0.4269; offset 0.4269 | 0.2987 | 8200000 |
| E3o_Pdata1 | gated pi=0.2 | 1.0 | 4: gain[axis 0] 0.4127; gain[axis 1] 0.4127; gain[axis 2] 0.4127; offset 0.4127 | 0.3567 | 9200000 |
| E3o_Pdata2 | gated pi=0.2 | 1.0 | 4: gain[axis 0] 0.4127; gain[axis 1] 0.4127; gain[axis 2] 0.4127; offset 0.4127 | 0.3567 | 10200000 |

Calibrated amplitudes (outcome-blind oracle R^2 about 50 +- 8 on development cohorts; method of `experiments/paper_calibrate.py`,
details and scans in `results/paper/CALIBRATION.md`, `results/cwk/CALIBRATION.json`). The cells share a key where they share an
amplitude (Pdata1, Pdata2 use the P key; Piid50 has its own key calibrated at iid fraction 0.5):

| key | sigma | gain | dev oracle % | verification oracle % | within 50 +- 8 | note |
|---|---|---|---|---|---|---|
| info_E1o_gated_p20_r3 | 0.8117 | 0.4101 | 50.0 | 45.7 | yes |  |
| force_unit_E1o | sigma_ref 0.8117 | force 0.7112 | - | - | - | unit = force / sigma_ref = 0.8762 |
| info_E1o_tanh_r3 | 0.6346 | 0.7958 | 49.4 | 45.4 | yes |  |
| info_E1o_gauss_r3 | 0.7648 | 0.5122 | 50.1 | 45.3 | yes |  |
| info_E1o_gated_p5_r3 | 0.8829 | 0.2550 | 50.5 | 49.8 | yes | refined (second round) |
| info_E1o_gated_p10_r3 | 0.8578 | 0.3097 | 50.2 | 45.9 | yes |  |
| info_E1o_gated_p50_r3 | 0.7256 | 0.5976 | 52.0 | 47.4 | yes |  |
| info_E1o_gated_p100_r3 | 0.6529 | 0.7559 | 49.6 | 45.5 | yes |  |
| info_E1o_gated_p20_r2 | 0.8 | 0.4356 | 50.2 | 51.2 | yes |  |
| info_E1o_gated_p20_r4 | 0.7747 | 0.4907 | 52.0 | 48.9 | yes |  |
| info_E1o_gated_p20_r3_iid50 | 0.8259 | 0.3792 | 50.2 | 44.6 | yes |  |
| info_E2o_gated_p20_r3 | 0.8 | 0.5600 | 50.9 | 54.6 | yes |  |
| force_unit_E2o | sigma_ref 0.8 | force 1.765 | - | - | - | unit = force / sigma_ref = 2.207 |
| info_E2o_tanh_r3 | 0.6229 | 1.0559 | 49.7 | 49.9 | yes |  |
| info_E2o_gauss_r3 | 0.76 | 0.6720 | 50.1 | 48.9 | yes |  |
| info_E3o_gated_p20_r4 | 0.8253 | 0.3567 | 50.1 | 57.6 | yes |  |
| info_E2o_gated_p5_r3 | 0.9229 | 0.2159 | 50.1 | 42.7 | yes | refined (second round) |
| force_unit_E3o | sigma_ref 0.8253 | force 1.452e+04 | - | - | - | unit = force / sigma_ref = 1.76e+04 |
| info_E2o_gated_p10_r3 | 0.8742 | 0.3522 | 51.1 | 47.1 | yes | refined (second round) |
| info_E2o_gated_p50_r3 | 0.7138 | 0.8014 | 50.1 | 52.1 | yes |  |
| info_E2o_gated_p100_r3 | 0.6424 | 1.0013 | 50.1 | 49.5 | yes |  |
| info_E3o_tanh_r4 | 0.6929 | 0.6270 | 49.1 | 51.0 | yes |  |
| info_E2o_gated_p20_r2 | 0.8205 | 0.5026 | 49.7 | 47.9 | yes | refined (second round) |
| info_E2o_gated_p20_r4 | 0.7775 | 0.6230 | 51.8 | 50.8 | yes |  |
| info_E2o_gated_p20_r3_iid50 | 0.8186 | 0.5079 | 50.6 | 53.8 | yes |  |
| info_E3o_gauss_r4 | 0.8173 | 0.3730 | 49.0 | 50.5 | yes |  |
| misalign_E2o_tanh | 0.6276 | 1.0427 | 50.1 | 47.4 | yes |  |
| info_E3o_gated_p5_r4 | 0.9199 | 0.1635 | 50.7 | 48.0 | yes | refined (second round) |
| misalign_E2o_tanh_nooff | 0.6185 | 1.0682 | 49.7 | 48.9 | yes |  |
| info_E3o_gated_p10_r4 | 0.8959 | 0.2125 | 50.1 | 50.8 | yes | refined (second round) |
| info_E3o_gated_p50_r4 | 0.7603 | 0.4894 | 51.2 | 56.4 | yes |  |
| info_E3o_gated_p100_r4 | 0.7093 | 0.5935 | 49.5 | 51.4 | yes |  |
| info_E3o_gated_p20_r3 | 0.8543 | 0.2975 | 49.7 | 54.7 | yes |  |
| info_E3o_gated_p20_r5 | 0.8 | 0.4083 | 50.3 | 57.7 | yes |  |
| info_E3o_gated_p20_r4_iid50 | 0.8537 | 0.2987 | 49.9 | 56.6 | yes |  |

### 2.3 MISALIGN (R5)

Ball-in-Cup, tanh law, iid fraction 1: MIS_E2o = thrust-misalignment driver (command perturbation s eps J x on the two actuated axes,
J = [[0, -1], [1, 0]], kind `rotate`) plus a common-offset driver (r = d = 2); MIS_E2o_nooff = the rotate driver alone (r = 1).

| cell | law | iid fraction | drivers r (channels / forces) | action gain | seed offset |
|---|---|---|---|---|---|
| MIS_E2o | tanh | 1.0 | 2: rotate 0.3138; offset 0.3138 | 1.0427 | 8600000 |
| MIS_E2o_nooff | tanh | 1.0 | 1: rotate 0.6185 | 1.0682 | 8600000 |

### 2.4 Upstream cells (R3)

The authors' own configurations, single offset driver (hand_l0 = their `hand`, cup, fetch), iid fraction 0.5 variants (hand_iid50, cup_iid50, fetch_iid50),
hand_load (body force), hand_affine50, hand_affine100 (gain fractions 0.5 and 1). Amplitudes, gains and population seeds are the authors' (no calibration).

### 2.5 Synthetic worlds

R4 (d = 1) and R5 (d = 2) worlds are defined in `experiments/paper_r4.py`, `experiments/paper_r5.py` (200k factual, 40k test, noise sd 0.05, uniform
drivers; seeds fixed in the scripts). The K0 generator (syn2, `cwk/eb.py: syn2_generate`) is used only by R0.

## 3. Arms (counterfactual R^2 on the primary group; * = privileged simulator information)

Every fit seed refits the mean predictor, the HH / EB / head models on 90% of the transitions (episode-wise split, seed-determined); the held-out 10%
is used only for selection (nugget, K, restarts, lambda, checkpoints). 5 fit seeds (3 for sweep and extra-data cells of R2b).

* trained, rr_learned (residual reuse with the learned mean), rr_exact* (exact means), oracle* (outcome-blind), copy*;
* hh_global, hh_binned (Stage 2 definitions: moment estimator on psi_j psi_k, Burer-Monteiro lift rank 4, ridge 1e-3, 1 or 8 k-means context bins);
* hh_binned_nug: hh_binned with a nugget N_i = c tr(C_i) / d, C_i = G_i G_i^T, c in {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} (c = 0 means 1e-6) chosen by the held-out
  factual Gaussian log-likelihood; transport G' (G^T G + N I)^-1 G^T r; hh_nug_ref*: c chosen on the development counterfactuals (privileged);
* hh_basis_{x, abs, x_x3, cubic}: one bin, psi = (1, x), (1, |x|), (1, x, x^3), (1, x, x^2, x^3) per actuated axis, nugget as hh_binned_nug, diagnostic cond(Phi^T Phi);
* rc_eb_gmm: HH lift (one bin, rank R = number of drivers of the cell, at least 2) + full-covariance Gaussian-mixture prior on xi by marginal likelihood
  (K in {16, 32}, 2 restarts each = best of 4 by held-out NLL, 3000 steps, Adam 1e-2 cosine, batch 1024), posterior mean in closed form;
* rc_npmle: same lift, 2000 grid atoms (regular grid for R <= 3, scrambled Sobol otherwise) on [-b, b]^R (b = 2 x the 99.5% quantile of the min-norm coefficients, in [3, 8]),
  EM 150 iterations on 20000 factual points, noise variance c tr(C)/d selected by held-out NLL over c in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1};
* rc_flow: same lift, RealNVP prior (6 affine couplings), trained by the importance-sampled marginal likelihood (1024 shared prior samples per step, 1500 steps),
  posterior mean by self-normalised importance sampling with 4096 prior samples;
* idt: independent-factor model (K = 4 location-mixture sources per driver, learned columns initialised at the HH lift, 3000 steps, 4 restarts for R <= 3 and 1 restart for R >= 4 because of the 4^R enumeration cost, held-out NLL);
  glr_iso: best-subset GLR isolation with the idt columns, lambda in {2, 4, 6, 8, 12} by the held-out factual NLL of the subset mixture; glr_iso_lamo*: lambda by the counterfactual loss (privileged);
* ls_signed_nug, cvae, pck_pol: as in Stage 2 (`results/cwk/STAGE2.md`);
* oracle_kernel* (exact mean, parent conditional covariance, two folds over draws), oracle_kernel_lm* (learned mean, parent map), oracle_quad* (learned mean, per-parent quadratic regression
  on the executed outcome; oracle_quad_ex*: exact mean, all pairwise products), oracle_struct_bayes* (per-parent finite-difference Jacobian (h = 0.5) in the drivers, exact
  Bayes over the world's driver laws by pattern enumeration and quadrature on the linearised map, likelihood noise = linearisation misfit; reports the linearisation R^2).

## 4. Seeds and sizes

Fit seeds 0-4 (R2b: 0-4 for P, T, G, Piid50, SEALED; 0-2 for pi5, pi10, pi50, pi100, rd, rd2, Pdata1, Pdata2). Data: 16384 episodes, 512 parents x 64 draws, 64 x 32 development.
R4 synthetic seeds 0-4, R5 synthetic seeds 0-2.

## 5. Metrics

Pooled counterfactual R^2 = 1 - mean loss / target variance over parents x draws x 3 queries (authors' definition); per-parent losses of every arm are saved
(`results/paper/raw/*.json`), per (parent, action) losses for R4. Held-out factual NLL (standardised units) for the HH / EB arms; cond(Phi^T Phi) for basis variants;
realised oracle R^2 and rho of every evaluation cell (calibration check).

## 6. Statistics

Whole-parent x fit-seed paired bootstrap, 2000 resamples (fixed seed 2026), 95% percentile intervals; paired differences use the same resamples; Holm across environments
where several environments test one hypothesis. Predictions are evaluated on point estimates with the rules in section 7 and reported with their intervals.

## 7. Pre-registered predictions (verbatim from the plan)

* P1 saturation: on Stage 2 cells, hh_binned_nug, rc_eb_gmm, oracle_quad* within 1.0 of oracle_kernel_lm*.
* P2 Gaussian control: |rc_eb_gmm - hh_binned_nug| <= 1.0 on G cells.
* P3 headroom: rc_eb_gmm - hh_binned_nug increases as pi decreases and as r - d increases; >= 0 at pi = 1 within CI.
* P4 sign: basis (1,|x|) matches (1,x) on held-out factual NLL within 0.01 nats and loses >= 10 points on cross-zero queries in hand_affine100.
* P5 no harm: on upstream additive cells every HH/EB arm within 1.5 of rr_learned.
* P6 rotation: HH lift below RR in the rotation world; within 1 of oracle in the misalignment world.

Evaluation rules as implemented in `experiments/paper_report.py` (fixed here): P1: |difference of pooled R^2| <= 1.0 for the three arms on each of the nine cells (verdict: all nine);
oracle_quad* uses the learned mean so that it is comparable with oracle_kernel_lm*. P2: the three G cells. P3: per environment, the sequence over pi in (0.05, 0.1, 0.2, 0.5, 1) is
non-increasing and over r - d in (0, 1, 2) non-decreasing in the point estimates, and the upper confidence limit at pi = 1 is >= 0. P4: held-out NLL difference (abs minus x) <= 0.01 nats in
absolute value and cross-zero R^2 loss of (1,|x|) relative to (1,x) >= 10 points, on hand_affine100 (cross-zero: some actuated axis changes sign between executed and query action).
P5: |arm - rr_learned| <= 1.5 for hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, rc_npmle, rc_flow, pck_pol on every upstream cell. P6: synthetic worlds of R5, rotation (I + x_1 J): hh_lift (R = 2)
below rr; misalignment + offset: |hh_lift (R = 3) - oracle| <= 1. The paper is written whatever the outcome.

## 8. Runs and order

R0 unit tests (done; `results/paper/R0.log`, `R0.md`: old cells bit-identical to the cached Stage 2 generator, gates shared across branches, rc_eb_gmm / idt / glr_iso / quadrature reproduce K0 exactly on g15, g30, tanh seed 0) -> R1 (`experiments/paper_r1.py`) || R2a (`paper_r2.py gen`) -> R2b (`paper_r2.py ladder`) -> R3 (`paper_r3.py`) -> R4, R5, R6
(`paper_r4.py synth|real`, `paper_r5.py synth|real`, `paper_r6.py`) -> R2c (`paper_r2.py validate E1o_P`, 64 parents); reports `paper_report.py r1 ... r6`.
Outputs `results/paper/R*.json|md`, raw per-seed files `results/paper/raw/`, logs `results/paper/logs/`.

R1: all nine cells, arms rr_learned, trained, hh_global, hh_binned, hh_binned_nug, hh_nug_ref*, oracle_kernel_lm*, oracle_quad*, rc_eb_gmm, pck_pol (+ basis variants on E1_nu50_sat_iid50).
R2b ladder arms: ls_signed_nug, hh_binned, hh_binned_nug, hh_nug_ref*, glr_iso, glr_iso_lamo*, rc_eb_gmm, rc_npmle (P, T, G, Piid50, SEALED only), rc_flow, cvae, idt, oracle_kernel_lm*, oracle_quad* (+ trained, rr_learned and the deterministic
oracle*, rr_exact*, oracle_kernel*, oracle_quad_ex*, oracle_struct_bayes*). R3: hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, rc_npmle, rc_flow, pck_pol, oracle_kernel_lm*. R4 real: hh_basis_x, hh_basis_abs,
oracle_kernel_lm* on hand_affine100 and E1_nu50_gain_iid100. R5 real: hh_global, hh_binned_nug, rc_eb_gmm, oracle_kernel_lm* on MIS_E2o, MIS_E2o_nooff. R6: iid100 cells, mean predictor trained on the first 512 episodes
(cross-fitted inside them, out-of-sample elsewhere), arms hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, oracle_kernel_lm*.
R2c: full-simulation quadrature (grid 11 per active driver, kernel width half a grid cell) vs oracle_struct_bayes* on E1o_P, 64 parents.

### Compute estimate

Measured on this machine (4 cores, 2 torch threads per heavy job; timings in `results/paper/logs/timing_*.json`, simulator timings from `cwk/worlds.py`):
one GMM fit (3000 steps) K = 32 / K = 16: 112 s / 58 s (d = 2, R = 3), 173 s / 84 s (d = 3, R = 4); RealNVP fit 36-43 s; NPMLE (5 noise levels x 150 EM iterations) 260-290 s; one idt fit (3000 steps, 4^R combinations) 128 s (R = 3), 644 s (R = 4), about 2600 s expected for R = 5 (hence idt restarts 4 for R <= 3 and 1 for R >= 4); mean predictor (2-fold + full) about 1.5-2 min; HH binned fits about 1 min per model.
Simulation per cell (E1o): factual collection 7 min, evaluation population with finite-difference Jacobians 10 min, development population 1.5 min (about 19 min per cell; E2o and E3o similar).

| item | seed-runs | minutes per seed-run | job-hours |
|---|---|---|---|
| R2a data (36 cells + sealed) and R3 data (cup, fetch and iid50 variants) | - | 19 per cell | about 13 simulator-hours |
| R1 (9 cells) | 45 | about 12 | 9 |
| R2b E1o and E2o (44 seed-runs each; rc_npmle only on P, T, G, Piid50) | 88 | about 22 | 33 |
| R2b E3o (R = 4, 5; idt 1 restart) | 44 | about 30 (rd2: 65) | 23 |
| R2b sealed cell | 5 | about 30 | 2.5 |
| R3 (9 upstream cells) | 45 | about 17 | 12.5 |
| R4 (synthetic 8 worlds x 5 seeds, 2 real cells x 5 seeds) | - | - | 2.3 |
| R5 (synthetic 3 seeds, 2 MISALIGN cells x 5 seeds) | - | - | 2.8 |
| R6 (3 cells x 5 seeds) | 15 | about 11 | 2.8 |
| R2c (64 parents) | - | - | 0.6 |
| **total** | | | **about 89 heavy job-hours plus about 13 simulator-hours** |

With two heavy jobs and one simulator job at a time this is about 48 hours of wall-clock time (the plan assumed about 45 job-hours). Levers applied: idt 1 restart for R >= 4; rc_npmle only on P, T, G, Piid50, SEALED in R2b; R3 stays at 5 seeds.


## 9. Sealed cell

After this protocol is committed, `python3 experiments/paper_sealed.py <protocol commit hash>` draws the environment (uniform over E1o, E2o, E3o), pi (log-uniform on [0.08, 0.6])
and an amplitude factor (U[0.85, 1.15] applied to the calibrated sigma interpolated in log pi over the calibrated gated cells) from an RNG seeded with the first 16 hexadecimal
digits of the hash, and writes `results/paper/sealed_cell.json` (hash, seed, every draw). The cell has r = d + 1, iid fraction 1 and seed offset 3000000. The hash is committed with the
file before any data of the cell are generated; no recalibration or inspection of its outcomes is allowed; its realised oracle R^2 is reported as it comes.

## 10. Deviations from the plan (disclosed before any run)

See the final report of phase 1 (listed also in `results/paper/CALIBRATION.md`): (1) rc_eb_gmm uses 2 restarts per K (best of 4 overall) instead of 4 per K, to keep the cost of R1 and R2b at
about half; the K0 reproduction in R0 uses the K0 setting (K = 32, 4 restarts). (2) oracle_quad* is defined with the learned mean (oracle_quad_ex* is the exact-mean variant of diag4.py).
(3) Gaussian control = gauss_clip with c = 3 (clipping probability 0.27% per driver). (4) glr_iso, glr_iso_lamo* and idt need the independent-factor columns and therefore the idt fit.
(5) idt restarts: 4 for R <= 3, 1 for R >= 4 (cost). (6) Calibration: keys whose verification cohort fell outside 50 +- 8 in the first round (E1o pi5; E2o pi5, pi10, rd; E3o pi5, pi10) were re-searched once on the verification cohort and verified on a third cohort (marked refined in the table; all end within the window). (7) rc_npmle is run in R2b only on the P, T, G, Piid50 and SEALED cells (not on the sweep and extra-data cells); R3 keeps 5 seeds. (8) Development cohorts are small (40 x 24), so the oracle R^2 of an evaluation population differs from the calibration target by sampling error of a few points; the realised value is reported. (13) R = number of drivers is given to the EB/idt lift (privileged structural information, applied identically to all EB arms); this makes explicit what item 5 of the lift rank already stated.
