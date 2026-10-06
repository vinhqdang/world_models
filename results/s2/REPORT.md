# S2 Phase B: selection-calibrated certified planning as a closed-loop wrapper

## Bottom line
SCC has no demonstrated decision benefit on this benchmark; it is a certification and diagnostic tool.
- Criterion (a), fall rate under shift with online update within +-0.02 of target: fails (0.146 at alpha 0.05, 0.181 at alpha 0.10).
- Criterion (b), success at equal fall at least 0.03 above the CVaR arm: holds nominally, but the gain is not attributable to the certificate: the fallback controller alone beats every planner arm and SCC, and random deferral at the same defer rate reproduces most of the gain over CVaR.

## Implementation (fixed in results/s2/config.json before any evaluation run)
Base planner: open-loop CEM, N=32, M=8, H=10, 3 iterations, chained CRN, es_s{0,1,2}; executed plan = mean of the final 3 elites. Certified quantity: open-loop 10-step pit-fall probability of the executed plan (not total cost, whose scale varies about 10x across states). Residual R = pT - pf, pT from 200 fresh true-simulator particles; q_hat from 4000 calibration states per model (wind 0.09, 70-76 base-planner episodes). Certificate U = pf + q_hat; commit iff U <= b = 0.10. Stage 1 never commits at b = 0.10 (q1 0.13-0.24 exceeds b), so every step escalates: top-4 candidates re-scored with 64 fresh independent particles (q2 0.03-0.06), commit if pf64 + q2 <= b, else defer to pi_0. Escalation costs +34% predictor rows (10320 vs 7680 per env step). pi_0: observed-position feedback law that steers to a fixed standoff above the pit, then to the goal; true-simulator check: success 0.998 and fall 0.002 at wind 0.09, 0.864 / 0.135 at wind 0.13; a faster variant gives 0.999 / 0.001 and 0.933 / 0.067. Deferral is therefore not costly here (pi_0 is safer than the planner and slower by about 6 steps). Online update (ACI): alpha_{t+1} = clip(alpha_t + 0.05 (alpha - err_t), 0.002, 0.5) on committed steps, with err_t from a counterfactual replay of the committed plan on the wind actually realised in the next H steps (verified to reproduce the direct outcome exactly on 4000 cases).

## Calibration validity (held-out, 40 repeats)
Executed-plan coverage, fall probability: alpha 0.05 nominal 0.95, selected band 0.952, pointwise 0.945; alpha 0.10 nominal 0.90, selected 0.901, pointwise 0.894. Total cost: 0.951 vs 0.935 and 0.898 vs 0.877. The model fall probability is badly optimistic (mean pf 0.007 vs mean true 0.04). For the argmin-selected pool member with chained CRN: pointwise coverage 0.915, 0.895, 0.859, 0.854 at pool sizes 2, 16, 32, 64 versus selected-band 0.904-0.911; for the elite-mean plan CEM executes there is no pointwise loss (0.905 vs 0.903). The collapse seen in Phase A (0.50 at N=32) is specific to independent-noise argmin selection.

## In distribution (wind 0.09; 288 episodes per main arm)
| arm | success | fall | timeout | rows per env step |
|---|---|---|---|---|
| base | 0.885 | 0.031 | 0.083 | 7680 |
| CVaR 0.5 | 0.785 | 0.021 | 0.194 | 7680 |
| SCC alpha 0.05 | 0.951 | 0.000 | 0.049 | 10320 |
| SCC alpha 0.10 | 0.948 | 0.000 | 0.052 | 10320 |
| base N=43 (equal compute, 144 episodes) | 0.903 | 0.007 | 0.090 | 10320 |
| random deferral p=0.22 | 0.948 | 0.038 | 0.014 | 7680 |
| pi_0 only | 1.000 | 0.000 | 0.000 | 0 |
False-commit rate P(commit and pT > 0.10): 0.012 (alpha 0.05) and 0.024 (alpha 0.10), below alpha; pointwise band 0.031 at N=32. SCC minus random deferral at matched defer rate: fall -0.038 [-0.062, -0.017] and -0.024 [-0.045, -0.007], success +0.003 and +0.014 (not significant). SCC minus pi_0 alone: success -0.049 [-0.076, -0.024].

## Shift (wind 0.09 to 0.13 after 200 steps, no retraining; post-shift episodes)
| arm | success | fall | timeout |
|---|---|---|---|
| base | 0.743 | 0.222 | 0.035 |
| CVaR 0.5 | 0.705 | 0.194 | 0.101 |
| failure-rate controller 0.05 | 0.715 | 0.194 | 0.090 |
| SCC 0.05 static | 0.861 | 0.104 | 0.035 |
| SCC 0.10 static | 0.875 | 0.094 | 0.031 |
| SCC 0.05 + online update | 0.823 | 0.146 | 0.031 |
| SCC 0.10 + online update | 0.781 | 0.181 | 0.038 |
| random deferral p=0.22 | 0.785 | 0.198 | 0.017 |
| pi_0 only | 0.878 | 0.122 | 0.000 |
| pi_0 fast only | 0.941 | 0.059 | 0.000 |
The static certificate loses validity under shift (false-commit rate 0.149 and 0.189, about three times alpha); the model's fall estimate does not see the wind change, so the defer rate does not move. The online update tracked its own replayed-fall target (0.059 at alpha 0.05) but that target does not control the episode fall rate, and it worsened false-commit rates (0.22-0.26) and success relative to static SCC (alpha 0.10: success -0.094 [-0.139, -0.049], fall +0.087 [+0.045, +0.132]). The failure-rate controller saturates again (final lambda 0.88-0.99). SCC minus random deferral: success +0.076 [+0.028, +0.125], fall -0.094 [-0.142, -0.045] (alpha 0.05). SCC is indistinguishable from pi_0 alone in success and fall, and slower.

## What is real, and what is not
Real: held-out coverage at the nominal level; in-distribution false-commit rate below alpha; the defer decision is selective (fewer falls than random deferral at equal defer rate, by 0.024-0.038 in distribution and 0.080-0.094 under shift). Not demonstrated: any success gain over always deferring to a hand-coded standoff controller; validity under shift; control of the episode fall rate by the online update; collapse of pointwise bands for the executed elite-mean plan. The information in the certificate comes from the 64-particle independent re-score, which shrinks the band 4-5 times.

## Caveats
Three model clusters (intervals resample episodes within model strata); partial pairing; time-correlated calibration states (70-76 episodes per model); deployment state distribution under SCC differs from calibration; calibration at wind 0.09 only; the benchmark is solved by a hand-coded standoff controller, which is the main confound; the online update was not run in distribution; b and pi_0 parameters were chosen on calibration data and one pilot seed; N=64 arms use 96 episodes and have no base arm; a fair decision-benefit test needs a benchmark where the fallback has a real cost.

## Files
selwm/s2_scc.py; experiments/s2_calib.py, s2_calfit.py, s2_eval.py, s2_analyze.py, s2_pi0_check.py, s2_argmin_diag.py, s2_sched.sh; results/s2/ (summary.txt|json, calib_N32*.txt, calib_N64*.txt, argmin_diag.txt, pi0_check.txt, config.json, runs/, pilot/).
