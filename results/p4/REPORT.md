# P4: open-ended gap hunt for a new world-model problem

Scope: about 95 literature queries (arXiv API, web search, about 45 abstract pages fetched, 2 full texts grepped) and two kill-test pilots on small simulators. Novelty probabilities are subjective; "not found" is weak evidence; statements about papers are abstract-level unless marked as full text.

## Bottom line
1. Nothing clears the bar. Best estimates: 0.10 for anchor-free sensor transfer with an identifiability certificate and for hierarchical subgoal selection under model exploitation; 0.04-0.08 for everything else.
2. The 2026 JEPA literature is saturated: velocity blind spot, distractors, latent-cost geometry, SIGReg variants, test-time adaptation, hierarchy, hidden parameters all have heuristic fixes.
3. Pilot A (sensor transfer through a frozen predictor) is a kill as an algorithm. At 250 target transitions (6 seeds) success is 0.49 and 3 of 6 seeds fail to recover the encoder; retraining on the same unlabelled transitions reaches 0.78 and regression on 50 paired anchors 0.74. The identifiability lemma held in a mechanism test, but MoVie (2307.00972, full text read) uses the same mechanism.
4. Pilot B (decision-focused data acquisition) is a kill: cross-regret reaches 0.150 against random 0.108, ensemble disagreement 0.088 and planner-distribution collection (DAgger style) 0.192, losing to the last by 0.031 on average and winning in 1 of 3 seeds. The limit is the smooth predictor's small residual leak at a dynamic discontinuity, which the planner exploits (PRISM-WM 2512.08411, ContactNets 2009.11193 are prior art).
5. Best honest paper: the analysis and negative-results paper (selection effects in sampling-based latent planning, coverage collapse of pointwise bands, invariance of the argmin to constant-width bands, certification without decision benefit when a good fallback exists) plus one mechanistic section on exploitation of small smoothing errors at dynamic discontinuities with a boundary-layer bound. Realistic venues: Neural Networks, Knowledge-Based Systems, Engineering Applications of AI, as an analysis paper.

## Problems considered (probability that a new-algorithm paper survives a prior-art check)
- Anchor-free sensor transfer for a frozen latent world model: 0.10. Gap: camera/background/viewpoint sensitivity (2506.09985, 2602.18639, 2608.05706). Nearest: MoVie 2307.00972, 2012.09811, NoMAD (Nature Communications 2025), JEPA-TTT 2610.00722, AdaJEPA 2606.32026, 2607.22430, 2306.06510. Sketch: exact consistency makes the solutions the commutant orbit of the frozen predictor, so the number of anchors needed is the number of free group parameters divided by d; with consistency error eps over k steps the encoder error is at most eps sum_{j<k} L^j.
- Decision-focused data acquisition: 0.06. Nearest: OPAX 2306.12371, 1203.1007, Plan2Explore-style disagreement; 2609.39235, 2607.10362, 2609.38383.
- Likelihood-free information-seeking planning for hidden context: 0.08 (not piloted). Nearest: 1910.08348, 1510.03591, 2202.07720, 2302.00171, 1808.00888, 2410.11234, 2508.20294, 2411.01342.
- Aliasing and memory without decoders: 0.07 (not piloted). Gap: 2610.04585, 2608.00591, 2609.32679. Nearest: 2608.20401, 2605.25313, 2601.14354, 2502.00466.
- Discontinuous dynamics and planner exploitation of smoothing: 0.05 as an algorithm; usable as an analysis finding. Nearest: 2512.08411, 2009.11193, 2512.10117, 2011.10605, 1803.02493.
- Timing, delays, irregular sampling: 0.04 (2403.12309, 2509.20869, 2607.27924, 2609.10464, 2609.35138, 2605.05951).
- Hierarchical subgoal selection under model exploitation: 0.10 (not piloted; continuation of the closed certification line, with a fallback that has a real cost). Gap: 2607.12547, 2406.00483. Nearest: 2006.11485, 2111.00213, 2604.03208, 2406.18053, 2606.24946.
- Latent cost and goal identifiability: 0.04 (2608.12959, 2608.29998, 2609.33030; ProWorld 2608.01926, 2610.01373, 2609.37441, 2607.25337, 2608.16287, 2601.00844).
- Not developed: certified symbolic abstraction (2303.12558, 2107.00110), 0.06; fragmented-trajectory multi-step learning (2610.02957, 2510.00739), 0.06.

## Pilot A: anchor-free sensor transfer (state-based point world, frozen JEPA trained on the source sensor, new encoder from unlabelled target transitions, CEM planning)
N = 250 (6 seeds), success: anchor-free consistency 0.49 (range 0.00-0.80), with sliced-Wasserstein matching 0.32, marginal matching only 0.02, 10 pairs 0.48, 50 pairs 0.74, 200 pairs 0.76, retrain on the same transitions 0.78; source-sensor reference 0.75. N = 1000 (3 seeds): 0.30, 0.48, 0.04, 0.56, 0.70, 0.74, 0.73. Mechanism test on a translation-invariant torus world: anchor-free adaptation recovers the source latents up to a translation (raw error 0.93-2.13, translation-aligned error 0.012-0.027); one paired anchor fixes it (raw error 0.004-0.067), as the commutant lemma predicts.

## Pilot B: decision-focused acquisition (2-D point robot with a blocking wall, ensemble of 4 MLPs, 1320 samples)
Success at 1320 samples: random 0.108, random in the task region 0.104, ensemble disagreement 0.088, cross-regret 0.150, planner-distribution collection 0.192; true model 0.74, no-wall model 0.11. With 24000 transitions and 120 epochs the blocked-transition error falls from 0.055 to 0.010 but planning success only from 0.10 to 0.24: the planner exploits the remaining leak of about 8% of a step.

## Caveats
Toy state-based worlds, 3-6 seeds, 60-80 episodes per cell, no multiplicity correction; Pilot B ensemble size and epochs were reduced during development; prior-art coverage is keyword-limited and some problems got fewer than 10 queries; withdrawn papers 2603.02263 and 2608.20065 must not be cited; the shared GPU session disappeared during the task, so all reported numbers are from CPU; ids that appear only in API listings or web results (2506.09985, 2603.19312, 2306.12371, 1203.1007, 1910.08348, 2303.12558, 2107.00110, 2403.03269) should be checked before citing.

## Files
results/p4/: common.py, pilot_a.py, pilot_a2.py, pilot_b.py, oracle_b.py, check_b_bigdata*.py, agg.py, pilot_a_s*.json, pilot_a2_s*.json, pilot_b_s*.json and logs.
