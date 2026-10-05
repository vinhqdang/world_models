# N2: removing sampling noise from the ranking (moment space, quadrature, sigma points, atoms, learned nodes)

Setup: `suite_hi/ckpt/es_s{0,1,2}.pt`, cliff_hi, open-loop CEM (N=32, H=10, 3 iterations), budget M=8 paths = 80 predictor evaluations per candidate. Ranking CIs are bootstrap over 72 scenarios (24 per model, 3 models, so only three model clusters); closed-loop CIs are bootstrap over episodes.

## Verdict
No defensible new-algorithm claim. Gaussian moment-space planning is worse than coupled Monte Carlo at equal predictor evaluations; sigma-point and Gauss-Hermite nodes fail on the learned saturating noise map; deterministic atom propagation matches but does not beat coupled Monte Carlo at equal compute. A balanced orthogonal-array-style sign design along the model's active noise direction (`dir_oa`) gives a small ranking gain over antithetic CRN that does not show in plan quality or closed-loop success. Learned shared nodes give larger ranking gains but are exploited by the optimiser (fixed-node curse). Every ingredient has prior art.

## Findings about the predictor
- The Jacobian of the one-step map with respect to the 8 noise coordinates has singular values (0.282, 0.010, 0.003) for model 0 (0.260/0.011/0.003 and 0.277/0.011/0.003 for models 1, 2): the noise map is effectively rank one.
- The dominant direction is shared across states (mean cosine to its average 0.92, 0.87, 0.997) and the response along it is a saturated step (y displacement at u = +-2 vbar is +-0.087; true wind amplitude 0.09). Fitted P(+) is 0.483, 0.479, 0.506. A one-dimensional function of the projection explains 93-97% of displacement variance at a fixed state. The learned law is effectively a two-atom mixture, which a Gaussian description cannot represent.
- Planner cost variance by Walsh order (1 / 2 / 3 / >=4): model 0 wide 0.764 / 0.193 / 0.027 / 0.016, local 0.689 / 0.255 / 0.036 / 0.020; model 1 wide 0.784 / 0.177 / 0.023 / 0.016, local 0.747 / 0.207 / 0.027 / 0.019. For candidate differences the low-order share is smaller (model 0 local 0.539 / 0.202 / 0.118 / 0.141).

## Theory (sketches)
- P1: E||Z-g||^2 = ||mu-g||^2 + tr Sigma, so a pure distance cost needs two moments; for dynamics affine in iid noise, any design with sample mean 0 and covariance I is exact. The learned map is saturating and the failure term depends on the whole law, so this does not close.
- P2: for a function on {-1,+1}^H, a randomly sign-flipped design has E err^2 = sum_{S != empty} gh_S^2 A_S^2 and the average of A_S^2 over S is at most 1/M for every design: no design beats iid Monte Carlo on average over functions; a strength-tau orthogonal array only moves alias weight to higher order. Predicted rms advantage of the design over antithetic MC for candidate differences: 0.81-0.85; measured centred-rmse ratio 1.05 (wide) and 0.94 (local).
- P3: atom-cloud propagation bias bound (delta_q + rho) sum_k L^k is vacuous for these models (1.2-1.8 position units against a Monte Carlo standard error of 0.094) because the learned map's Lipschitz constant dominates.
- P4: a fixed node set is a deterministic function of the plan that the optimiser can exploit (sample-average-approximation overfit); merge decisions in atom propagation are discontinuous in the plan.

## Ranking benchmark (reference: 512-path scrambled-Sobol antithetic; reference-versus-reference Spearman 0.989 wide, 0.928 local)
Local regime: indep Spearman 0.353, crn 0.813, crn_anti 0.837, dir_oa 0.855, learned 0.886, moment 0.714, atoms5_fit 0.771, ut3 0.711, atoms11_fit (2x compute) 0.816. Regret (reference cost of the picked plan minus best): indep 0.744, crn_anti 0.141, dir_oa 0.120, learned 0.109, atoms5_fit 0.185.
Wide regime: indep 0.717, crn 0.909, crn_anti 0.926, dir_oa 0.937, learned 0.960, moment 0.844, atoms5_fit 0.940, ut3 0.846, atoms11_fit 0.944. Regret: indep 0.671, crn_anti 0.142, dir_oa 0.109, learned 0.060.
Paired gains of dir_oa over crn_anti: wide Spearman +0.011 [+0.006, +0.017], regret -0.033 [-0.054, -0.014]; local Spearman +0.018 [+0.006, +0.032], regret -0.021 [-0.041, -0.004]. Full paired tables in `rank_summary.md`.

## CEM-level test (3 models x 64 states x 3 runs, plan scored by an independent 1024-path reference)
Reference cost: indep 3.645, crn_anti 2.503, crn_chain3 2.696, dir_oa 2.476, learned 2.525, atoms5_fit 2.628. dir_oa minus crn_anti -0.027 [-0.077, +0.023] (indistinguishable); learned minus dir_oa +0.049 [+0.006, +0.094] (the learned nodes lose their advantage once the optimiser searches against them); atoms5_fit +0.125 worse. The sticky/antithetic confound for crn_chain3 is not separated by this test.

## Closed loop (open loop, equal compute; E=16, 260 steps per run, eval seeds 500-502)
indep 141 episodes success 0.738, fall 0.085, timeout 0.177; crn_chain3 161 episodes 0.839 / 0.031 / 0.130; dir_oa 142 episodes 0.803 / 0.028 / 0.169. Paired: crn_chain3 vs indep success +0.076 (p=0.076); dir_oa vs indep +0.063 (p=0.24); dir_oa vs crn_chain3 -0.007 (p=1.0). About 150 episodes per arm, below the 300 requested; a 0.03 difference is not resolvable. To extend: append lines `SCHEME MODEL EVALSEED STEPS` to `results/n2/jobs.txt`, delete `results/n2/STOP`, run `experiments/n2_queue.sh`, then `python experiments/n2_analyze.py cl`.

## Novelty by component
Moment-space belief planning (PILCO, U-MPPI 2306.12369, FORESEE 2209.12644, 2608.02519): known, loses here. Sigma-point nodes: known, fail on saturating maps. Atom expansion with merge compression (FORESEE 2209.12644, Stochastic MuZero chance nodes, Gaussian-sum filters): known skeleton, no equal-compute gain. Shared nodes, antithetic, QMC (PEGASUS 1301.3878, 2605.04732): known. Orthogonal-array sign designs (Owen 1992, 1994): known statistics, small gain. Learned node sets (2405.15059, 2510.03745, 2608.09335, 2501.19045): known idea class, fails under optimisation here. Plan-conditioned amortised node placement: not tested.
Directions not refuted: (a) randomised, robust learned node distribution trained against the optimiser's selection; (b) training the predictor so its noise map is a K-atom law with known weights (close to Stochastic MuZero chance codes).

## Caveats
Three model seeds, one benchmark; the reference is itself a 512-path estimate (regret differences below about 0.03 not resolvable); learned nodes trained on model 0 only; the local-regime offset range was changed once after a smoke test and before the main run; no scheme was tuned on evaluation seeds. `rank_main.log` was truncated by a stale background run; `rank_main.json` is intact.

## Files
Code: `selwm/n2_model.py`, `experiments/n2_*.py`, `n2_queue.sh`, `n2_chain.sh`. Results: `results/n2/` (rank_*.json, rank_summary.md, walsh_m*.json, atomerr.json, diag_noise_map.json, learned_nodes_q.pt, cemeval.json, cemeval_summary.md, cl_*.json, cl_summary.md).
