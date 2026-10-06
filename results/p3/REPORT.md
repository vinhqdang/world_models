# P3: evaluating world models without the environment (predictive validity and model selection)

## Verdict
Maybe-to-dead as a Q1 algorithm paper. The problem is real, but at least four papers from August and September 2026 already attack exactly the question of which offline metric predicts planning success, OPE-style estimators predate them, and in the pilot (29 models, fixed planner) no candidate beat the architecture-family prior on success. One exploratory signal predicts the fall rate.

## Prior art (32 arXiv-API queries and 12 web searches; every id fetched; table in PRIOR_ART.md)
2609.32322 (decision-relevant prediction error; 55 models; rho -0.84 against total error -0.25), 2607.01736 (40 offline validation metrics against CEM-MPC return, no environment access), 2608.12959, 2608.10145, 2606.15032, 2609.33030, 2608.29998, 2609.39235; policies ranked by world models (2506.00613, 2505.19017); offline-MBRL model selection (2502.11480, 2111.14346); OPE-proxy model selection (2007.09055, 2302.00141, 2008.04990, 2502.08021, 2103.16596); model-based OPE, doubly robust, MAGIC, FQE, replay (2104.13877, 1511.03722, 1604.00923, 1903.08738, 1003.5956). No planner-conditioned offline validity estimator with a finite-sample ranking guarantee for latent world models found (weak evidence).

## Candidates (all statistics from a fixed held-out behaviour log; the simulator is never stepped)
1. Plan-pool replay validation: replay estimator on logged plan pools plus a union bound; novelty 0.08.
2. Occupancy-weighted validation with a direct-method baseline: importance-weighted cross-validation plus the simulation lemma; 0.08.
3. Selected-plan optimism (from the selection theory in paper/theory.tex): mean of realised minus predicted planner cost over the plans the model picks from sub-pools; regret at most 2 E max_i |e_i|; 0.15.
4. Hindsight plan ranking: run on 2 models only (det_s0 mean rank 0.29, es_s0 0.43; the safer stochastic model ranks worse); 0.12.

## Pilot
29 models on StochNav cliff_hi: the 9 suite_hi checkpoints plus 20 new models (det, gauss, es at different training lengths and three low-coverage variants), identity encoder in every model so latents are comparable (encoder-quality effects untested). Ground truth: one fixed weak planner (open-loop CEM, N=16, M=4, 2 iterations, H=10), eval seed 901, 99-265 episodes per model; split-half reliability Kendall tau 0.67 for success and 0.47 for fall. Success by family: det 0.651, gauss 0.518, es 0.414; fall: det 0.320, gauss 0.186, es 0.237 (weak-planner regime; the stronger D8 planner gives es success about 0.83).
Success target (Kendall tau over 29 models): replay pool pick gain +0.62 [+0.43, +0.78], direct-method imagined success +0.46, multi-step MSE +0.08, one-step MSE +0.06, NLL -0.31, one-step energy score -0.67, occupancy-weighted energy score -0.56, selected-plan optimism -0.49. The replay win is a family confound: a kind-only oracle that knows only the architecture family scores +0.75, and within each family no statistic has an interval excluding zero. Top-1 selection by replay or direct method picks a det model (success 0.61-0.62, regret 0.11-0.12 against 0.728), the same as the family prior; one-step energy score and NLL pick es_s1 (success 0.449, fall 0.316).
Fall target (tau with minus fall): selected-plan optimism +0.67 [+0.45, +0.85], within-kind +0.53 [+0.25, +0.79]; multi-step energy score +0.35; one-step energy score +0.16; NLL +0.22; imagined fall -0.42. Selected-plan optimism beats the best baseline by +0.32 (paired [+0.02, +0.59]) and one-step energy score by +0.51 [+0.28, +0.76]. Caveats: exploratory, about 20 statistics times 2 targets examined, the statistic highlighted after seeing the table (direction fixed beforehand); it anti-predicts success (-0.49), and selecting by it picks the safest, lowest-success model (success 0.17). Longer-trained models are sharper, more committed, and fall more.
Other: held-out NLL and energy score rank success inversely to the truth here; occupancy weighting is no better than uniform one-step energy score; planner-cost RMS over logged plans has no signal.

## Missing
Replication of the fall signal on a held-out set of models and with the stronger planner; encoder diversity (pixel-encoder suite); the ranking guarantee is not instantiated (replay is restricted to logged support); one environment and one planner; the latent-distance-correlation baseline is degenerate with the identity encoder and was not computed; the GPU session disappeared before downloads.

## Files
RESULTS_TABLE.md, SELECTION.md, PRIOR_ART.md, correlations.json, gt/, stats/, stats2/, ckpt/; selwm/p3_offline.py; experiments/p3_*.
