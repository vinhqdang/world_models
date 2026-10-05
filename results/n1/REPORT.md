# N1: cross-fitted selection for CEM planning with stochastic latent world models

## Verdict
No defensible new-algorithm claim. The planner suggested by the selection-bias theory (propose on noise fold A, re-score a short list on a fresh fold B, execute the fold-B winner) is train/validation splitting plus a Rinott-style clean-up stage; its guarantee is a union bound over the short list with fresh data, a mild generalisation of Thm `verify` in `paper/theory.tex`. The plan-quality effect is large on the true simulator; in closed loop it is small and only partly significant. The largest gain in this family is the chained common-random-number coupling, not the cross-fit.

## Algorithm (selwm/n1_cf.py)
CF-CEM with N=32, T=3 iterations, M_A=7 particles in fold A, M_B=24 in fold B, short list of 4 (mean of the 3 final elites plus the 3 elites). Fold A may use independent noise or chained CRN. Fold B is fresh iid noise common to the short list. Equal compute: 3*32*7 + 4*24 = 768 particle-rollouts per environment step, same as the baseline 3*32*8. The difference J_B - J_A of the fold-A winner is an unbiased estimate of the planner's in-sample optimism at every replan.

## Theory
Theorem 1 (cross-fit validity; known ingredients). With S fold-A-measurable and fold B independent: (i) E[J_B(a) | fold A] = J(a) on S; (ii) with probability at least 1-delta, J(selected) <= min_S J + 2 R_D sqrt(ln(2K/delta)/(2 M_B)), R_D the range of cost differences over S; (iii) the bound does not depend on N, T, M_A or the coupling in fold A; (iv) J_B - J_A of the fold-A winner is an unbiased optimism meter. The split does not reduce ranking noise and does not help when the best plan is outside S.
Proposition 2 (noise-stability form of the winner's curse). With u_B = rho u_A + sqrt(1-rho^2) xi, Mehler's formula gives E[J_B(a) | fold A] - J(a) = (1/M) sum_m sum_{k>=1} rho^k c_k(a, u_{A,m}); the retained optimism is R(rho) = sum_k rho^k b_k with R(0)=0 and R(1)=1.

## Results
Mechanism on the true simulator (1000 states, 1024 fresh draws per plan): cf_crn3 J 1.061 versus ol_crn3 1.103 (difference -0.042 [-0.048, -0.035]), fail probability 0.155 versus 0.175; it recovers about 59% of the cost headroom to an 8x-compute reference at equal compute. Allocation matters: M_A/M_B = 4/96 is worse than the baseline. On the learned model (model seed 0 only) the difference is not significant.

Closed loop, simulator (E=100, 6 seeds): success ol_indep 0.787, ol_crn3 0.860, cf_crn3 0.871, cf_indep 0.833. cf_crn3 minus ol_crn3: success +0.010 (p=0.092), fall +0.003, timeout -0.013 (p=0.0018). cf_indep minus ol_indep: success +0.044 (p=3e-9), fall -0.021.

Closed loop, learned energy-score models (E=16, three model seeds): success ol_indep 0.760 (325 episodes), ol_crn3 0.847 (339), cf_crn3 0.893 (363). cf_crn3 minus ol_indep: +0.131 (paired p=4e-6). cf_crn3 minus ol_crn3: +0.046 unpaired [-0.003, +0.095], paired +0.037 (p=0.14), not significant. ol_crn3 minus ol_indep: +0.094 (p=6e-4).

Dose-response (simulator, 1000 states): gain over ol_crn3 decays monotonically as the folds are made dependent (rho = 0, 0.5, 0.8, 0.95, 1: -0.0415, -0.0404, -0.0379, -0.0331, -0.0091), and the retained optimism follows the Mehler curve (-0.02, 0.16, 0.37, 0.57, 1.00). The optimism meter reads 0.130 against a true 0.127 for cf_crn3. In-sample optimism is about 3x smaller under CRN than under independent noise (0.13 versus 0.36).

## Limitations
Small closed-loop effect over the matched CRN baseline; partial pairing and episodes treated as independent within a model; three learned models give no model-level interval; learned-model mechanism diagnostic covers one model seed; cf_indep not run in closed loop on learned models; the (7, 24) split was chosen a priori and compared on a small grid on a diagnostic seed only; adaptive allocation from the optimism meter not implemented; episode counts differ across arms (325/339/363).

## Prior art (not new)
Winner's curse and sample-split repairs (Smith and Winkler 2006; van Hasselt 2010; arXiv 1302.7175, 1509.06461, 2605.18887); two-stage selection with fresh second-stage samples (Dudewicz and Dalal 1975; Rinott 1978; Boesel, Nelson and Kim 2003); high-confidence policy improvement with held-out test (Thomas et al. 2015); adaptive analysis with a holdout (arXiv 1411.2664); independent evaluator re-scoring a CEM shortlist (arXiv 2610.00921); debiasing the optimizer's curse (arXiv 2107.12438, 2306.10081); model exploitation (arXiv 2605.15960). Not found: cross-fitting or a fresh-fold decision inside CEM/MPPI with a stochastic latent predictor (weak evidence).

## Sharpest honest framing
A diagnosis plus a cheap guard: scenario-set overfitting of CRN planners, measured by an unbiased free optimism meter and removed by a fresh-fold decision at the same compute, with the Mehler dose-response as the quantitative checkable claim; closed-loop effect reported as small against the CRN baseline.

## Files
Code: selwm/n1_cf.py, experiments/n1_common.py, n1_eval.py, n1_diag.py, n1_diag_summ.py, n1_analyze.py and run scripts. Results: results/n1/oracle, results/n1/learned, analysis tables and diag_* files in results/n1/.
