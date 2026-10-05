# Planning with stochastic latent world models: measurements so far

All numbers below come from the scripts in `experiments/` (CPU runs, E=16 parallel environments, N=32 candidates, M=8 particles,
horizon 10, 3 CEM iterations, 3 model seeds, 300 to 520 environment steps per run, i.e. 150 to 500 episodes per configuration).
Intervals are bootstrap 95% over model seeds (3 seeds, so they are wide and should not be over-read).

Benchmark: `cliff_hi` in `selwm/stochnav.py` (point agent, absorbing pit between start and goal, bimodal wind of amplitude 0.09,
agent step 0.05). World models use a frozen encoder on (x, y, fallen); planning cost is squared latent distance to the goal,
plus an exemplar-based failure penalty in latent space (same for all planners).
True simulator with the same planner (upper bound): success 0.889, fall 0.040 at wind 0.09; success 0.774, fall 0.068 at wind 0.13.

## Main comparison (in distribution)

| planner | success | fall | timeout |
|---|---|---|---|
| deterministic predictor, mean rollout (LeWM-style) | 0.669 | 0.331 | 0.000 |
| ensemble of 3 deterministic predictors | 0.689 | 0.306 | 0.004 |
| Gaussian head, mean rollout | 0.460 | 0.540 | 0.000 |
| Gaussian head, particles, expected cost | 0.781 | 0.093 | 0.127 |
| energy-score predictor, particles, expected cost | 0.827 | 0.079 | 0.094 |
| energy-score predictor, CVaR mix 0.5 | 0.688 | 0.076 | 0.237 |
| energy-score predictor, CVaR 1.0 | 0.609 | 0.088 | 0.303 |
| energy-score + empirical-Bayes shrinkage | 0.768 | 0.065 | 0.168 |
| energy-score + racing + shrinkage + CVaR 0.5 | 0.772 | 0.072 | 0.156 |
| energy-score + common random numbers + shrinkage + CVaR 0.5 | 0.724 | 0.025 | 0.251 |

Reading: almost all of the gain comes from using a predictive distribution and averaging the cost over particles.
Risk-averse scoring, shrinkage, racing and common random numbers did not raise success over plain expected cost.

## Dynamics shift (wind 0.09 to 0.13 after 200 steps, models not retrained)

| planner | success after shift | fall after shift |
|---|---|---|
| deterministic, mean rollout | 0.399 | 0.601 |
| energy-score, expected cost | 0.709 | 0.200 |
| energy-score, CVaR 0.5 | 0.656 | 0.216 |
| energy-score, online failure-rate controller (target 0.05) | 0.615 | 0.220 (controller saturates at lam 0.92) |
| energy-score + online scalar spread recalibration | 0.665 | 0.240 (estimated scale 1.37, true 1.44) |
| energy-score + oracle scalar spread scale 1.44 | 0.518 | 0.303 |
| energy-score + per-dimension spread recalibration | 0.599 | 0.197 (estimated y scale 1.35, true 1.44) |
| online fine-tuning of the predictor (MSE / NLL / energy score), lr 3e-4 | 0.461 / 0.547 / 0.550 | 0.535 / 0.266 / 0.263 |

None of the adaptation schemes tried improved on the unadapted energy-score planner. Online fine-tuning also degraded the
pre-shift performance (energy score: success 0.717, fall 0.234 before the shift).

## Other measurements

- Learned encoders with SIGReg (pixel and state inputs, 10k steps, small networks): Spearman correlation between squared latent
  distance and true distance to the goal 0.44 to 0.76, for latent sizes 4 to 64. Frozen identity encoder: 1.00.
- Noise-space importance sampling (gradient or cross-entropy shifted Gaussian proposal) for failure probability of fixed candidate
  plans: RMSE was higher than plain Monte Carlo for gradient shifts (0.21 to 4.3 vs 0.10); see `experiments/is_microbench.py`.
- Tail blindness amplified by selection: P(select a risky candidate) = 1 - (1 - F_Bin(k*; M, p))^{N_r}, checked by simulation.
