# N3: candidate novel algorithms for world-model planning

Scope: literature scan (about 30 arXiv-API queries, about 25 web queries; "not found" is weak evidence) and one small CPU diagnostic. All cited arXiv ids were fetched and the titles checked (`refs.md`). Novelty probabilities are subjective estimates that an informed reviewer cannot point to a paper that already does the core step.

## Bottom line
1. No candidate clears the novelty bar with high confidence; the best sit at 25-28%.
2. Strongest new checkable fact (StochNav, learned energy-score models, `selection_check.py`): a pointwise-calibrated pessimism band loses coverage for the *selected* plan as the candidate pool grows. Selected-plan coverage at nominal 0.90 for N = 2, 4, 8, 16, 32, 64 is 0.84, 0.74, 0.56, 0.39, 0.23, 0.10 (second checkpoint: 0.82 down to 0.09). A band calibrated on the selected plan, or on the pool maximum, keeps 0.87-0.95 at every N.
3. Top pick: selection-calibrated certified planning (SCC): N-indexed calibration of the selected plan; commit / escalate / defer with a finite-sample false-commit guarantee; the exact collapse law for pointwise bands.
4. Second pick: ExoCRN, a stochastic predictor with action-independent exogenous noise so that common random numbers is a valid counterfactual coupling (unconstrained predictors do not identify the noise map).
5. Constant-width bands cannot change an argmin (same invariance as the monotone-shrinkage proposition). In the diagnostic, beta x standard error did not reduce plan-level regret (0.427 to 0.422 / 0.429 at N = 64); selecting with 8x more particles did (0.427 to 0.136). Selection error here is a selection-variance problem, not a missing-penalty problem.
6. LeWM TwoRoom has little headroom for selection fixes: success saturates in N; predicted latent progress is 6-10x too optimistic but rank-correlated; the bottleneck is objective and horizon.

## Open problems stated by 2025-2026 papers (closest to our theory)
- Model exploitation unavoidable on large plan sets, with a "safe horizon" only for finite MDPs (2605.15960).
- Larger proposal pools lower feasibility of the selected plan; heuristic fix ASAR (2607.23602, 2609.24745).
- Scoring errors change both the decision and later CEM proposals (2610.00921).
- Latent conformal bounds are pointwise or tube-based (2606.15594, 2602.12047, 2609.34300, 2505.00779) and are not selection-valid.
- Deterministic predictors average multimodal futures (2512.24497, 2601.14354).

## Diagnostic (plan-level winner's curse and coverage)
cliff_hi, checkpoints es_s0 and es_s1, H = 10, 240 states (120 calibration, 120 test), pool of 64 open-loop candidates per state, J8 = score with 8 particles, true cost from 200 simulator particles, 30 sub-pool resamples per state. Coverage = fraction of test states where the true cost of the selected plan is at most J8 plus the band.

| N | regret vs best in pool, select by J8 / J64 | coverage pointwise / selected / pool-max (es_s0) | width pointwise / selected / pool-max |
|---|---|---|---|
| 2 | 0.092 / 0.031 | 0.84 / 0.91 / 0.92 | 0.23 / 0.51 / 0.53 |
| 4 | 0.169 / 0.055 | 0.74 / 0.92 / 0.93 | 0.21 / 0.78 / 0.82 |
| 8 | 0.240 / 0.077 | 0.56 / 0.90 / 0.92 | 0.21 / 0.92 / 0.99 |
| 16 | 0.292 / 0.093 | 0.39 / 0.89 / 0.92 | 0.22 / 1.06 / 1.17 |
| 32 | 0.357 / 0.115 | 0.23 / 0.89 / 0.94 | 0.22 / 1.17 / 1.41 |
| 64 | 0.427 / 0.136 | 0.10 / 0.93 / 0.95 | 0.23 / 1.41 / 1.57 |

For i.i.d. equal-truth candidates the exact pointwise coverage is (1-alpha)^N (0.81, 0.66, 0.43, 0.19, 0.03, 0.001); the measured decay is slower because residuals are heterogeneous and correlated.

Caveats: random pools rather than CEM-refined (mean true failure probability of pool members 0.51); open-loop plan cost, not closed-loop outcome; two checkpoints and one state sample; the RMS of J8 minus J64 is large (2.4 and 3.5), suggesting heavy-tailed particles. Must be re-run on real CEM populations before any claim.

## Ideas ranked
| Rank | Idea | Novelty prob. | Main risk |
|---|---|---|---|
| 1 | SCC: selection-calibrated certified planning | 0.25 | "conformal on a planner output"; decision benefit unproven; may weaken on CEM pools |
| 2 | ExoCRN: identifiable exogenous noise for valid CRN | 0.28 | narrow scope; earlier CRN arms with an unidentified coupling were null |
| 3 | Thresholdout-CEM: information-budgeted adaptive search | 0.20 | overlaps cross-fitting direction |
| 4 | Energy-distance ambiguity planning, pool-uniform radius | 0.20 | quadratic cost outside RKHS; classical kernel DRO |
| 5 | CNAS: certified near-optimal action sets | 0.18 | ASAR plus conformal; ad hoc tie-break |
| 6 | Execution-consistent scoring of feedback plans | 0.18 | incremental on our own method |
| 7 | OSC: budget-indexed order-statistic calibration | 0.15 | exchangeability fails for CEM pools |
| 8 | Planner-tail proper score | 0.12 | 2607.01171 |

Assembly desk-reject risk is estimated at 30-40% even with the right framing; the honest framing for SCC is a certification/calibration contribution with a theorem and a demonstrated failure of standard practice, not a new planner.

## SCC: formal statement
States are exchangeable draws from the deployment distribution. A pipeline P = (pool generator G_N, scorer j with M particles, selection rule) maps (s, internal randomness) to a selected plan a* and score j(s, a*). L(s, a) is the realised cost in the true environment; R_i = L(s_i, a*_i) - j(s_i, a*_i) on n calibration states; q_hat is the ceil((1-alpha)(n+1))-th smallest R_i.
(a) U(s) = j(s, a*) + q_hat covers: P(L(s_{n+1}, a*_{n+1}) <= U) >= 1 - alpha for each fixed pipeline (split conformal).
(b) Commit when U <= fallback bound; lambda chosen by conformal risk control so the false-commit rate is at most alpha.
(c) A pointwise band calibrated on random pool members has selected-plan coverage at most (1-alpha)^N in the i.i.d. equal-truth case and decays with N in general; the pool-max band keeps N-independent coverage.
(d) Under drift, adaptive conformal updates on the residual stream give long-run false-commit rate within a vanishing gap of alpha.
Non-claim: SCC raises success. The claim is controlled false-commit and fall rate at a quantified timeout cost.

Protocol. Phase A (coverage on real CEM final populations and iteration-1 populations, N by sub-sampling, 500 calibration and 500 test states per checkpoint; kill if pointwise coverage at N = 32 is within 5 points of 0.90). Phase B (closed loop, D8 protocol, arms: open loop, CVaR mix, failure-rate controller, SCC; in distribution and under wind shift with and without adaptive updates; continue only if fall rate is within 0.02 of target under shift and success at equal fall is at least 0.03 above the CVaR arm). Phase C (public LeWM TwoRoom: calibrate predicted versus realised progress, report defer rate on offset goals).

## ExoCRN: formal statement
Environment s_{t+1} = F(s_t, a_t) + eps_t, eps_t independent of a_t given s_t. Learn z' = m(z, a) + g(u | z) with energy score plus an independence penalty (energy distance between the residual law within action bins and the pooled residual law).
Theorem A: under realisability and zero penalty, g has the law of eps minus E[eps | z]; the abduced noise is identified up to a measure-preserving map, so a shared noise sequence is a structural counterfactual coupling.
Theorem B: paired-difference variance (1/M)[Var c_a + Var c_b - 2 Cov(c_a, c_b)]; Cov >= 0 when both costs are coordinate-wise monotone in the noise; misselection against an incumbent at most N exp(-2 M Delta^2 / r_pair^2); the sign condition fails for plans that hedge in opposite directions around a hazard.
Protocol: identifiability test (correlation of abduced noise with the true wind sign across actions at one state), ranking test (Kendall tau and top-1 regret versus M for independent, unconstrained-CRN and structured-CRN; kill if no gain over unconstrained CRN at M <= 4), closed loop at equal compute including the gust variant.

## Recommendation
Frame the algorithmic claim as the certification layer (SCC) on top of the feedback-parameterised stochastic planner; use ExoCRN only if its ranking test shows a clear gain. Run Phase A on real CEM pools first and apply the kill criterion. If both kill criteria trigger, frame the paper as analysis (selection theory plus negative results). Do not claim as new: feedback parameterisation, conformal tubes, pessimism, or CRN in themselves.
