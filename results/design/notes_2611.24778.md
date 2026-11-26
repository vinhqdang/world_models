# Reading notes: Counterfactual World Models: JEPAs That Reason About "What If"

Balestriero, LeCun, Varaku; arXiv 2611.24778 v1, 18 November 2026 (preprint; 78 pages, main text pages 1-31 read in full, appendices not read).

## Problem
Individual counterfactual: after an observed transition (z_t, a_t, z_{t+1}) produced by a hidden disturbance eps, predict the outcome under an alternative action a' with the SAME eps. Distinct from intervention (averages over eps).

## Non-identification results
- Prop 2.1: factual (marginal) laws do not identify the coupling between potential outcomes.
- Thm 2.2 (gauge non-identification): for push-forward generative models P_{s,a} = (T_{s,a})# nu, any nu-preserving bijections G_{s,a} give T_{s,a} o G_{s,a} with identical factual laws but different counterfactual couplings T_{s,a'} o G_{s,a'} o G_{s,a}^{-1} o T_{s,a}^{-1}. Factual likelihood cannot identify the same-disturbance counterfactual.
- Thm H.2 / Cor H.3: positive minimax risk for unrestricted individual counterfactuals.

## Method (residual reuse)
z_hat_{t+1}(a') = z_{t+1} + mu(H_t, a') - mu(H_t, a_t) (Def. 3.1, eq. 6); readout variant in physical coordinates (eq. 7). No counterfactual data, no extra parameters, no retraining.
- Exact iff the action contrast does not depend on eps; exact for additive responses F = m(z, a) + q(z, eps) (eq. 11); Prop 3.2: the only rule exact in every additive world.
- Error identity: E||T_R - Y(a')||^2 = tr Var(Delta), Delta = individual action contrast (eq. 10); with a fitted predictor add ||e(z, a') - e(z, a)||^2 (eq. 16).
- Thm 3.3: tr Var(Delta) <= K^2 ||a' - a||^2 tr Cov(eps) under a mixed-derivative bound K; interaction ratio rho = tr Var(Delta) / tr Var(Y(a')) <= (K ||a' - a|| / kappa)^2; residual reuse beats the best outcome-blind predictor iff rho < 1.
- Thm 3.4 / Cor 3.5: separate action and disturbance channels over a transition of duration h give F_h = z + h v(z, a) + h w(z, eps) + R_h with ||R_h|| <= C h^2, hence error <= 2 C h^2 and rho = O(h^2). Separate channels are ASSUMED, not derived from smoothness.
- Thm 3.6: minimax rate-optimal with sharp leading constant (4 C^2 h^4 / (1 + C^2 h^2) <= minimax <= 4 C^2 h^4).
- Thm 3.7 / Cor 3.8 / Prop 3.9: squared loss on a frozen encoder identifies the conditional mean; with shared inputs and independent randomisation it equals the interventional mean (no propensity weighting needed).
- Factual data give only a lower bound on the interaction (squared Wasserstein / reverse triangle); certifying small interaction needs paired outcomes or knowledge of where the disturbance enters.

## Stated failure modes (Table 1, Table 2, Sec. 3.3 limits, Sec. 4.3)
- Multiplicative (gain) faults: interaction is first order; rho near 2 at every duration; learned correction falls to 6.4-11.0% R^2 (below trained predictors at 41-42%).
- Contacts, clipping, saturation: Push-T block velocities have rho > 1; residual reuse with exact simulator means gives R^2 of 5.6, -15.5 and -44.4% (below outcome-blind references at 20-39%). Failure is present before any visual predictor is fitted.
- Folded coordinates: cos(pitch) rho = 1.67 while sin(pitch) is additive.
- Raw pixels do not transport (rho about 2 even at short durations); MAE features are not additive in their own metric (rho = 1.92).
- One residual over one transition only; no chaining over long horizons; residual helps later decisions only when the disturbance persists (Sec. 4.4).
- Causal logged actions assumed.

## Experiments
Seven rendered simulators (HandReach, Ball-in-Cup, FetchReach, PassWater, Racetrack, Push-T, BlueROV2); bounded actuator disturbances; frozen DINOv2-S/B, MAE-B (plus DINOv3, SigLIP 2, CLIP, EVA-02); predictor with two attention blocks of width 1024 on k = 1-3 frame embeddings; 8k factual episodes. Physical counterfactual R^2 after readout 85.6-95.7% vs 38.1-45.6% trained predictor vs 49.1-51.1% oracle without outcome; feature-space correction 73.4-94.1%. BlueROV2 70.7-75.5% at 8k episodes, 90.1-93.5% with 16x data. Hindsight planning success 35-69% vs 9-25% trained predictor vs 14-23% true-simulator planner without the outcome; two retries 51-97%. Persistent fault over three transitions: +4-25 to 11-51 points; fresh disturbances: -4 to +6; no disturbance: replanning success drops 4-15 points.

## Authors' open directions (Sec. 5)
(i) paired trials to certify where residual reuse is safe; (ii) train an encoder or readout so actions and disturbances add; (iii) transport rules that switch with the contact mode or rescale the residual with the command (contacts and gain faults); (iv) a predictor-selection criterion targeted at the action contrast rather than full-feature factual error; (v) chaining counterfactuals over several transitions.

## Relevance to this project
- Pre-empts the "abduce the realised exogenous noise and reuse it" ideas (ExoCRN in results/n3; the conditioning step of Doob-ledger planning in results/design/B).
- Its gauge theorem is the formal version of our observation that the noise map of an unconstrained stochastic predictor is not identified (results/n2).
- Its central failure mode (contacts / discontinuities) coincides with our wall-world exploitation result (results/p4 B.3; results/design/A, C). Directions (iii) and (v) are explicitly listed as open by the authors, so any work there races against them.
