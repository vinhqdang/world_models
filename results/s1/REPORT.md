# S1 Phase A: selection-calibrated certified planning, kill test on real CEM populations

Setup: cliff_hi, es_s{0,1,2}, H=10, M=8, CEM N=32 with 3 iterations (final and iteration-1 populations) and a separate CEM-64 run; 1000 states per checkpoint (25% start states, 75% mid-path states near the pit); planner score J8, re-score J64, two independent 200-particle true-simulator costs (one for residuals, one for a cross-fitted regret oracle); pools of size N sub-sampled 20 times per state; selection by argmin J8; nominal level 0.90; five random 500/500 calibration/test splits per checkpoint; pooled results use 1500 calibration and 1500 test states.

## Verdict
The kill criterion does not trigger, so the direction survives Phase A, but more weakly than the random-pool diagnostic suggested.

Final CEM-32 population, pooled (selected-plan coverage at nominal 0.90):
| N | pointwise band | selected-plan conformal | pool-max | optimism of selected plan | true cost of selected plan |
|---|---|---|---|---|---|
| 2 | 0.859 | 0.903 | 0.921 | -0.858 | 2.062 |
| 4 | 0.789 | 0.902 | 0.931 | -0.347 | 1.984 |
| 8 | 0.697 | 0.903 | 0.935 | -0.012 | 1.944 |
| 16 | 0.594 | 0.902 | 0.937 | +0.236 | 1.928 |
| 32 | 0.500 [0.47, 0.53] | 0.902 [0.89, 0.92] | 0.942 [0.93, 0.95] | +0.423 | 1.933 |
CEM-64 final population, N=64: pointwise 0.411 [0.39, 0.44], selected-plan 0.903, pool-max 0.949. Iteration-1 population at N=32: pointwise 0.492.

## Observations
1. The pointwise collapse is real but milder than the random-pool diagnostic (0.23 at N=32 there, 0.50 here); candidates in a CEM population have correlated residuals. Validity of the selected-plan band holds by split-conformal construction and is not itself a finding.
2. The certified band is expensive: width grows from 0.45 (N=2) to 1.24 (N=32) while the true cost of the selected plan stops improving after N of about 8.
3. The selection error is mostly particle noise: at N=32 the particle-noise part (J64 minus J8 at the selected plan) is +1.74 and the model-bias part (true minus J64) is -1.31. Selecting with J64 instead of J8 cuts cross-fitted regret from 0.27 to 0.11.
4. Chained CRN (one checkpoint, es_s0): pointwise coverage at N=32 is 0.813 against 0.500 without; the effect shrinks but the criterion would not trigger.

## Commit/defer extra (final population, pooled, N=32)
Commit iff J8 of the selected plan is at most a threshold chosen by conformal risk control on P(commit and bad), bad meaning true pit-fall probability above b. Selected-plan calibration holds the false-commit rate at or below alpha at every N (0.041-0.049 for alpha 0.05; 0.087-0.100 for alpha 0.10); a threshold calibrated on random pool members rises with N, to 0.29 at N=32 (about six times the target) for alpha 0.05, b 0.1. Commit rates are low (10-14% at alpha 0.05; 16-31% at alpha 0.10) and the probability of a bad plan given commit is still 0.33-0.55 at b 0.1-0.2. A Learn-then-Test variant controlling P(bad | commit) commits nothing, because J8 alone does not separate plans.

## Caveats
CEM returns the mean of the elites, not argmin J8 from the final population, so the harm to the deployed planner may be overstated; open loop only on a synthetic state mix; one definition of pointwise baseline (marginal band); checkpoints share a recipe; CRN covers one checkpoint; the five splits overlap.

## Files
selwm/s1_pop.py, experiments/s1_collect.py, s1_analyze.py, s1_commit.py; results/s1/phaseA_*.txt|json, commit_*.txt|json, pops_*.npz.
