# Design A: learned models of discontinuous dynamics under an optimising planner

## Verdict of the designer
No design reaches novelty 0.3. Pre-checks falsified two of three candidates and the planner half of the third. What survives (a graph-sup training rule and a sup-graph calibration certificate) attaches to Design C's planner; neither is a standalone algorithm.

## Pre-checks (exploratory, CPU, not pre-registered; wall world of results/p4/pilot_b.py, 24,000 uniform transitions, 4-MLP ensemble, CEM N=100, H=14; 80-200 tasks, SE 0.035-0.05)
| arm | success | tasks |
|---|---|---|
| nominal learned model, N=100 / N=800 | 0.205 / 0.205 | 200 |
| true model, N=100 / N=800 | 0.57 / 0.82 | 200 |
| supervised "blocked" hard gate, threshold 0.5, N=100 / N=800 | 0.43 / 0.665 | 200 |
| same gate, threshold 0.1 / 0.9 | 0.42 / 0.35 | 200 |
| residual-magnitude penalty, lambda 0.3 / 3 | 0.35 / 0.00 | 120 |
| ensemble-disagreement penalty, lambda 1 / 10 | 0.008 / 0.00 | 120 |
| input-neighbourhood adversary, greedy stencil, rho_s 0.02 / 0.04 | 0.32 / 0.275 | 200 |
| PGD adversary, input ball / output ball 0.1 / 0.3 step | 0.06-0.075 / 0.163 / 0.037 | 80 |
| tearing map (candidate 3) | 0.115 | 200 |
| label-free residual peeling, hard gate / two-branch worst case | 0.185 / 0.28 | 200 |
Readings: the exploit set has order-one mass (the learned model is flat from N=100 to 800 while the true model gains 0.25); blocked moves are predicted to creep about 8% of a step and the model tunnels within about 0.012 of the wall; on held-out transitions the output residual max is 0.86 step but allowing an input shift of 0.05 drops it to 0.147 (graph residual); graph-sup training lowered mean residual (0.076/0.065 to 0.041/0.037) and graph-residual max (1.02/0.86 to 0.275/0.269) on an under-trained MSE arm (suggestive only); sampled, greedy and gradient adversaries fail because the inner maximum is itself a planning problem with jumps, whereas Design C's exact box propagation worked; supervised hard gating closes most of the gap; label-free mode discovery fails (residual flags catch 59% of blocked samples, 58% false positives).

## Prior art that changes the picture
- 2608.17956 (August 2026) already reports the planner pinned at a mode boundary by a mode-blind model and proves a Lipschitz localization budget: the exploitation finding and the boundary-layer bound are not unpublished as analysis.
- 2606.03521: MPC with PGD adversarial rollouts on a learned model (output ball).
- 2512.09929: world models trained on adversarial latent and action input perturbations (max over the ball).

## Candidate 1 (recommended as a merge with Design C): graph-enclosure world model (GEM)
Measure, train and certify model error in directed graph (Hausdorff) distance with input slack instead of output norm: the only sup-type error that is non-vacuous at a jump.
Definition: Delta_hat is a (rho, eps)-graph enclosure if for every x there is x' with d(x, x') <= rho and ||Delta_hat(x') - Delta(x)|| <= eps.
Training: 25% of epochs MSE warm start; then per sample K=8 uniform shifts in U(rho_tr) plus zero, e_ik = ||Delta_hat(x_i + d_ik) - Delta_i||^2 / sigma_s^2, l_i = -tau log mean_k exp(-e_ik / tau) with tau = 0.05^2; loss = mean l_i + mean of the top 10% of l_i; rho_tr = (0.25, 0); Adam 3e-3, cosine, batch 256.
Calibration (20% held-out): grid over rho_s in {0, ..., 0.6}, rho_a in {0, 0.1, 0.2}; r_i(rho) = min over 256 shifts plus 5 projected gradient steps; feasible pairs with max_i r_i <= 0.1; pick the feasible pair with smallest median inflation.
Planner: robust cost over pseudo-orbits computed by sound set propagation (Design C's box propagation); not sampled or gradient adversaries.
Theorems: T1 an accurate-outside-band model is a (c r, eps0 + L c r) enclosure (elementary); T2 uniform enclosure from a calibration set with covering radius eta (covering argument; cursed in dimension); T3 J <= J_rob for all plans, so the executed plan's true cost is non-increasing in the pool (elementary); T4 any valid output-norm tube around a continuous model needs radius at least J/2 everywhere; T5 "no free band" (1-D sketch); T6 minimax lower bound (conjecture).
Prior art: 2606.03521, 2512.09929, 2608.17956, 2606.15594, 2602.12047, 2409.08249, 2512.08411, 2011.10605, 1803.02493, 2009.11193, 2310.12972, 2204.12581, 2005.13239, 2102.08363, 2005.05951, 2301.13375, 2003.08938, 2206.02829, 1810.10775, Bertsimas-Nohadani-Teo 2010 (doi 10.1287/opre.1090.0715), 2407.11478, 2602.06864, 2009.12409, 1401.5200, Sendov 1990, 2105.09095, 2607.23602, 2605.15960, 2610.00921, 2401.17899. Novelty 0.20 standalone, about 0.25 merged with Design C. Biggest risk: read as robust MPC plus an errors-in-variables loss; a tuned conservative gate matches it.
Kill test: Design C's protocol (E1 wall-stay, E3 door, E4 pusher; 600 paired tasks per arm) plus arms KR-WM with graph-sup base against MSE base, sup-graph radius against the 0.995-quantile, supervised hard gate, residual penalty grid, output-ball propagation, N sweep 100 / 400 / 1600. Pass: (i) graph-sup base lowers the calibrated radius by at least 25% and raises KR-WM success by at least 0.05 pooled over E1, E3 (CI excludes 0); (ii) at least 0.80 of true-model success at matched N; (iii) at least 0.08 above the best tuned cheap baseline including the supervised gate; (iv) success at N=1600 not below N=100 minus 0.02; (v) J_rob >= J_true on at least 99% of executed plans. About 4 CPU-hours.

## Candidate 2: two-branch gate (novelty 0.12; killed label-free)
Hard-gated point model; worse of two branches on the gate's ambiguity set. Label-free flags are wrong (0.185); the branch rule adds about 0.095; a supervised gate reaches 0.43 / 0.665. Useful as baseline and ablation.

## Candidate 3: tearing map (killed)
Discontinuous model f(x + d(x)) with a regressed shift field: oracle shifts cut the residual max from 0.90 to 0.24 step, but the learned field leaves held-out error unchanged and planning falls to 0.115 (the argmin shift is non-unique).

## Recommendation of the designer
Design C is stronger on the planner side; Design A adds the graph-sup training loss, a uniform sup-graph certificate, the two-branch ablation and the negative results. Run one kill test of KR-WM with graph-sup base and sup-graph calibration; criterion (i) decides whether A contributes. Any paper must cite 2608.17956.

Pre-check scripts were in the session scratchpad (A/chk2.py to chk11.py, dg.py), not in the repository.
