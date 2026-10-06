# P3 prior art (all ids fetched through the arXiv API id_list endpoint and titles matched; 2026 papers additionally opened with WebFetch)

Queries: 32 arXiv-API keyword queries (relevance sort; weak recall) plus 12 web searches. Nearest papers:

| id | title | relation |
|---|---|---|
| 2609.32322 | Not All Errors Matter: Decision-Relevant Prediction Error Predicts Planning Quality | closest: 55 models, DRPE vs total error (rho -0.84 vs -0.25), gridworld, requires known decision-relevant dimensions; sufficient-condition result |
| 2607.01736 | Predicting Closed-Loop Performance of Latent World Models: Offline Checkpoint Selection ... LunarLander | 40 offline validation metrics vs CEM-MPC return, no environment access; structural metrics (ROF/CROF); single environment |
| 2608.12959 | The Objective Is the Bottleneck | 4 checkpoints, success rank-orders with cost-metric quality and inversely with one-step accuracy |
| 2608.10145 | The Evaluation Protocol Determines the Result (LeWM TwoRoom reproduction) | one-step accuracy does not predict long-horizon success |
| 2606.15032 | How Should World Models Be Evaluated for Embodied Decision-Making? | position paper: policy-ranking agreement, model exploitability, calibration |
| 2609.33030 | What Must a World Model Distinguish for Planning? | sufficiency hierarchy (mechanism, response, decision) |
| 2608.29998 | The Intervention Gap in Latent World Models | matched-intervention audit; needs environment |
| 2609.39235 | The Planning Limits of Latent World Models | effective-horizon limit |
| 2506.00613, 2505.19017 | WorldGym, WorldEval | world model as evaluator of policies (MMRV, Pearson) -- the reverse direction |
| 2502.11480 | Active Model Selection for Offline MBRL (BOMS) | model selection with a few online evaluations |
| 2111.14346 | Pessimistic Model Selection for Offline Deep RL | offline selection with guarantee (value-function based) |
| 2007.09055, 2107.11003, 2110.14000, 2205.08716 | offline hyperparameter / model selection with OPE proxies (FQE best) | |
| 2302.00141, 2008.04990, 2502.08021 | Bellman-error / BVFT / LSTD-Tournament selection | value-function selection |
| 2104.13877 | Autoregressive Dynamics Models for OPE | model-based OPE, ranking metrics |
| 2103.16596, 1911.06854 | DOPE benchmark, empirical OPE study | rank correlation / regret@1 protocol |
| 1511.03722, 1604.00923, 1903.08738 | doubly robust, MAGIC, FQE | |
| 1003.5956 | Li et al. replay evaluator | unbiased offline evaluation by matching logged choices (the pool-replay statistic here is a plan-level instance) |
| 2003.00030, 1806.01265, 2306.17366, 2002.04523 | policy-aware / value-aware / lambda-models / objective mismatch | decision-aware training, not validation |
