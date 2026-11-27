# K0 pre-kill: pre-registered protocol (written before any run)

Purpose: test whether the independent-factor (IDT) transport beats the dependent random-coefficient empirical-Bayes baseline (rc_eb_gmm) on the synthetic overcomplete generator.

## Generator (syn2)
d = 2 outputs, r = 3 drivers. Common offset plus one gain per actuator, J = [[1, 0.3], [-0.2, 0.8]], G(x) = J [ (1,-1)^T | (x1,0)^T | (0,x2)^T ]. x ~ U[-1,1]^2. 80,000 factual samples, 20,000 test samples, noise sd 0.05, one action x per factual sample, target = G(x') xi at a fresh x' with the same xi and no noise. Drivers: independent, mean 0, variance 1.
Settings: (a) gated pi = 0.15, (b) gated pi = 0.30 (xi_j = Bern(pi) * U(-1,1) / sqrt(pi/3)), (c) tanh (xi_j = tanh(2 N(0,1)) / 0.78). Seeds 0-4 per setting. All arms use identical data per seed.

## Arms
1. idt: learned independent-factor Bayes, G = psi(x)^T G_q with psi = (1, x), HH initialisation, K = 4 location-mixture sources per driver (means initialised on linspace(-1.2, 1.2), re-standardised to zero mean and unit variance at each step), R = 3, diagonal noise with floor 1e-3, Adam lr 1e-3 cosine, 3000 steps, batch 512, grad-clip 5, best checkpoint (every 500 steps) by held-out NLL, best of 4 restarts by held-out factual NLL on a random 10% split of the factual data. Posterior-mean transport.
2. rc_eb_gmm: HH lift + full-covariance GMM prior on xi, K = 32, 3000 steps, Adam lr 1e-2 cosine, batch 1024, best of 4 restarts by held-out factual NLL (same 10% split). Posterior-mean transport.
3. hh_blup: linear BLUP with the HH lift (nugget max(sigma^2, 1e-3)).
4. oracle_blup*: linear BLUP with true G and true noise variance.
5. exact_bayes*: exact posterior mean with true G and true prior. Gated: quadrature (pattern enumeration over which drivers are active; midpoint grid of spacing <= 0.03-0.04 in the free coordinates, remaining coordinate integrated analytically with truncated-normal moments when the common-offset column is active). Tanh: 3-D midpoint grid in driver space with the exact tanh-Gaussian density (quadrature reference; resolution check reported).
6. glr_iso: best-subset (GLR) fault isolation using the learned idt columns, penalty lambda chosen from {2,4,6,8,12} by held-out factual NLL of the subset-mixture model implied by the penalty (prior weight proportional to exp(-lambda |S|/2), plug-in least-squares amplitudes, learned noise).
Held-out split: the first fit uses 90% of the factual data (HH lift, restarts, all fits); the 10% split is used only for selection.

## Metric
Counterfactual R^2 (%) = 100 (1 - sum ||pred - target||^2 / sum ||target - mean(target)||^2) on the 20,000 test points.

## Decision
Per setting: mean over the 5 seeds of (idt - rc_eb_gmm) with a 95% t-interval over the 5 paired seeds, and a paired bootstrap over test points pooled across seeds (2000 replicates; each replicate resamples test indices within every seed, the same indices for both arms, recomputes per-seed R^2, averages the difference). K0 FAILS (NO-GO) if, in 2 or more of the 3 settings, the mean difference is < +1.5 points or its CI includes 0. Otherwise K0 PASSES. A setting counts as failing if either its t-interval or its bootstrap interval includes 0 or its mean is below +1.5 (the more demanding reading).
## Sanity
Learned arms must not exceed exact_bayes* by more than its CI (t half-width over seeds); flagged otherwise.
## Allowed deviations (to be documented in K0.md)
Bug fixes only if needed to run (e.g. the 4000-particle reference is replaced by quadrature). Closed-form 2x2 inverses replace Cholesky for speed (identical math). Nothing in this protocol is changed after results are seen.
