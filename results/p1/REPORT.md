# P1: action sensitivity of action-conditioned world models

## Verdict
Dead as a new-algorithm direction. About 15 papers from May to October 2026 cover the problem (identifiability and excitation theory, remedies for every fix considered, planning-ranking diagnostics). In a confounded state-based pilot no simple or algorithmic fix beat plain maximum likelihood reliably, and budget-matched dithering beat every one of them by 5 to 20 times: the problem is limited by information, not by the training loss. The GPU was not used.

## Prior art (about 70 arXiv-API queries and 9 web searches; every id fetched; table in prior_art.md)
Inverse-dynamics losses (2609.30264, 2606.31232, 2610.03137), action contrastive losses (2608.17542, 2606.07304), inverse-dynamics cycle consistency in the planning cost (2607.02403), transition separation (2607.26712), advantage readout (2608.06706), action-invariant head with orthogonality (2607.18715), counterfactual consistency (2608.04653, 2609.37378, 2609.22816), intervention alignment (2608.24885), active queries (2609.23753, 2604.01985). Theory and diagnostics: identifiability and a counterfactual-error amplification governed by a conditional action-excitation margin (2607.22430), causal-state identifiability (2609.31161), action identifiability via Bayes inverse risk (2606.09028), frozen JEPA models rank candidate actions badly (2609.05461), decision-sufficiency hierarchy (2609.33030), irreducible action-specific error of non-conditioned predictors (2608.22197). Causal-ML prior art: regularisation-induced confounding (1706.09523), neural treatment-effect estimators (1906.02120), copycat problem (2207.09705). Only unexplored slot: the fully observational regime with propensity or orthogonalisation applied to latent world models (weak evidence; the idea is textbook in causal ML).

## Candidates
Orthogonal action-effect learning (novelty 0.08-0.12), excitation-aware active acquisition (about 0.10), randomisation-inference certificate (about 0.12; close to the exhausted certification direction).
Facts: planning regret is bounded by the oscillation over candidate plans of the model-cost error, so only the action-dependent error matters; in a partially linear model no estimator beats Fisher information N sigma^2 / sigma_w^2 (sigma the behaviour-action noise, sigma_w the process noise); an inverse-dynamics head that sees the state attains the same loss for an action-ignoring predictor up to the unexplained action variance.

## Pilot (selwm/p1_env.py; dev seeds 900-902 for the auxiliary weight only; eval seeds 0-9)
4-dimensional state, 2-dimensional action, s' = 0.9 s + 0.3 sin(Rs) + 0.3 G(s) a + noise; behaviour a = clip(K s) + sigma noise, about 97% predictable from the state; open-loop CEM, horizon 6; regret normalised by the zero-action plan cost. sigma = 0.05, N = 3000:
| method | normalised regret | paired d vs mle | wins |
|---|---|---|---|
| oracle interventional data | 0.013 | -0.276 | 10/10 |
| mix 0.25 (budget-matched dithering) | 0.025 | -0.264 | 10/10 |
| mix 0.10 | 0.041 | -0.248 | 10/10 |
| mix 0.05 | 0.060 | -0.229 | 10/10 |
| affine action head (structural prior) | 0.215 | -0.074 | 10/10 |
| mle | 0.289 | | |
| inverse-dynamics on displacement | 0.289 | +0.001 | 8/10 |
| inverse-dynamics | 0.290 | +0.002 | 2/10 |
| action NCE | 0.329 | +0.041 | 0/10 |
| residual inverse-dynamics | 0.336 | +0.048 | 2/10 |
| centred action input | 0.354 | +0.065 | 2/10 |
| orthogonalised effect loss | 0.561 | +0.272 | 1/10 |
Excitation sweep (sigma = 0.02, 0.05, 0.15, 0.4): mle regret 0.750, 0.289, 0.161, 0.051; centred input 0.515, 0.354, 0.162, 0.048 (sign flips with sigma); dithering mix 0.10 stays at 0.03-0.04. The sensitivity ratio is 0.9-1.0 for every method: the error is in the direction and gain of the action effect, not in ignoring the action.

## Limits
State-based, one environment family, three dev seeds (the centred-input gain on dev reversed on eval); no learned encoder, so latent-collapse effects are untested; open-loop regret only.

## Recommendation
Drop as the main algorithmic direction; do not claim action sensitivity, counterfactual, inverse-dynamics, contrastive or advantage-readout losses as new.
