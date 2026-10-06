# P2: compounding error and long-horizon rollout drift in latent world models

## Verdict
Dead as a new-algorithm direction (under 10% that any candidate survives a Q1 novelty check). Every mechanism derived has a close 2025-2026 or classical precedent, and the pilot found no effect of the best candidate against simple baselines. One narrow residual: on chaotic systems, trajectory accuracy and long-run statistics conflict.

## Prior art (about 30 arXiv-API queries and 15 web searches; every id fetched; logs scan1-3.txt, abstracts abs1-5.txt)
Finite-time Lyapunov defect propagation and predictability horizon (2606.13092), drift-aware calibrated re-sensing deadline (2607.01537), off-manifold Gramian and contraction shield (2607.10362), multi-horizon consistency (2607.21645), spectrally constrained latent core with rollout-error bound (2607.19719), stationary-law-preserving transitions (2609.32657), transverse exponential stability (2609.17901), geometry-aware noise injection (2509.20201), relabelling off-attractor rollouts (2609.32864), rollout-decoded reconstruction (2608.25017), phase-drift correction (2608.07189), variable-length latent models (2606.21775), semigroup consistency (2605.26324), compression, Koopman and noise injection baselines (2609.30198), invariant-measure objective (DySLIM 2402.04467), dissipative projection (2410.00976), generalized teacher forcing (2306.04406), video exposure-bias fixes (2512.12080, 2509.25161, 2607.20368, 2609.35491), effective-rank regularisation (2607.27036), rollout truncation from uncertainty (2609.21482), DART (1703.09327), multi-step loss (2402.03146), PDE-Refiner (2308.05732), parareal with a learned coarse model (1912.05958), and Data-as-Demonstrator (2015, not on arXiv).

## Candidates
C1 split-error rule (normal contraction plus self-calibrated normal-direction noise): about 5%; it is DART plus geometry-aware noise plus normal-hyperbolicity; piloted. C2 semigroup-defect step control: 10-12%; the defect cannot see a bias shared by both predictors; not piloted. C3 tangent-linear Gramian-weighted one-step loss: about 5%; equals multi-step backpropagation to first order; smoke test only.

## Pilot (Lorenz-63, Van der Pol; 16-dimensional smooth random embedding; same residual MLP; 5 test seeds; hyperparameters tuned on separate validation seeds)
Valid prediction time (steps until RMS error exceeds 0.3 standardised units), Lorenz: teacher forcing 9.4, isotropic noise 16.0, normal-only noise 15.7, scheduled sampling 25.3, self-calibrated normal noise 11.3 (noisy-oracle reference 38.9). Blow-up fraction of long chains: teacher forcing 1.00, isotropic 0.21, normal 0.25, scheduled sampling 0.73, self-calibrated 0.01; energy distance to the true attractor: 2.7, 1.8, 3.1, 0.22. Van der Pol VPT: teacher forcing 42, isotropic 228, normal 451, scheduled sampling 267, self-calibrated 436 (intervals overlap heavily; heavy-tailed).
Compute-matched controls (10000 steps, 4 seeds): Lorenz self-calibrated minus isotropic -7.4 +- 3.0, minus normal -7.5 +- 2.3, minus scheduled sampling -24.8 +- 2.5 (significantly worse); Van der Pol differences have intervals containing zero. Autoencoder projection hurts VPT on both systems but removes blow-ups in noise-trained models. The self-calibration fixed point is unstable on Lorenz (sigma oscillates between 0.3 and 0.025) and sits at its floor on Van der Pol.
Only qualitative finding: on Lorenz, scheduled sampling has the best VPT (35 steps at 10k) but about 80% of long chains blow up, while self-calibrated noise and isotropic noise plus projection have the best statistics (blow-up 0-0.03, energy distance 0.13-0.6) at about half the VPT; no arm is good on both (DySLIM, 2609.32657, 2410.00976 territory).

## Residual gap
A provable trade-off between trajectory validity and invariant-measure fidelity under expanding dynamics (VPT capped near log(1/eps)/lambda, no predictor reaching the cap and preserving the stationary law without explicit measure control); about 10-15% novelty and an analysis-style paper.

## Limitations
Hyperparameters tuned on one validation seed; scheduled-sampling optimum at the edge of its grid on Lorenz; short Lorenz VPT; no image observations or control-cost metric; fixed training budgets; the Gramian-weighted loss has no proper evaluation.

## Files
selwm/p2_dyn.py, p2_train.py; experiments/p2_*.py; results/p2/ (tables.md, tables_matched.md, raw jsonl, scan and abstract logs).
