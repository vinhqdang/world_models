# D9: feedback-parameterised CEM x coupled noise (crn_chain3), M = 8 particles

## What was built (new files only)
- `selwm/d9_model.py`: `CoupledLatentFB` (subclass of d6 `CoupledLatentModel`). `rollout_fb` draws one noise tensor `u_all` (E, Nb, M, H, 8) from the d6 `NoiseSource` (Nb = 1 shared across candidates for crn/crn_chain3, Nb = N for indep). The same `u_all[..., t]` drives the sampled path z_t; the nominal path zbar_t uses u = 0; `a_t = clip(a_nom + K (z_t - zbar_t)[:2])`. Checked: K = 0 reproduces the d6 open-loop rollout exactly (max abs diff 0.0).
- `experiments/d9_eval.py` (flags `--fb none|const|step`, `--scheme indep|crn|crn_chain3`, `--M`, ...). It calls `begin_replan` with the episode-reset mask each step (as d6_eval), uses d3 `fb_cem` unchanged, stores per-episode outcomes and counts predictor forward rows. `experiments/d9_run.sh` (queue), `d9_summ.py`, `d9_pair.py`, `d9_sweep.py` (tables), `d9_diag.py` (offline diagnostic).
- Protocol for all arms: ckpts `suite_hi/ckpt/es_s{0,1,2}.pt`, variant cliff_hi, evaluation seed 211 (env seed 5211; earlier results/ used seeds 0-9), E=16, N=32, M=8, H=10, iters=3, 450 steps, ep_len 120, kappa 3, stage_w 1, lam 0, kmax 7.5, `--fb step`.

Commands: `experiments/d9_run.sh "ol_indep:none:indep fbstep_crn3:step:crn_chain3" "0 1 2"` and `experiments/d9_run.sh "fbstep_indep:step:indep ol_crn3:none:crn_chain3" "0 1 2"` (two parallel queues, OMP/MKL threads = 1); then `python experiments/d9_summ.py`, `python experiments/d9_pair.py`. Sweep: `d9_run.sh "<arms>" "<seeds>" <M> _M<M>` then `python experiments/d9_sweep.py`. Diagnostic: `python experiments/d9_diag.py`.

## Main result (pooled over 3 model seeds; bootstrap 95% CI over episodes; `summary.txt`)
| arm | episodes | success | fall | timeout | median steps |
|---|---|---|---|---|---|
| open loop, indep (baseline) | 288 | 0.757 [0.708,0.806] | 0.101 [0.069,0.139] | 0.142 [0.104,0.184] | 59.0 [54,65] |
| open loop, crn_chain3 | 329 | 0.888 [0.851,0.921] | 0.033 [0.015,0.055] | 0.079 [0.052,0.109] | 51.0 [49,54] |
| feedback step, indep | 377 | 0.902 [0.873,0.931] | 0.069 [0.045,0.095] | 0.029 [0.013,0.048] | 48.0 [45,50] |
| feedback step + crn_chain3 | 398 | 0.925 [0.897,0.950] | 0.043 [0.025,0.065] | 0.033 [0.018,0.050] | 43.5 [41,46] |

Per model seed (success/fall/timeout, episodes): see `summary.txt`. Combined arm: m0 0.845/0.086/0.069 (116), m1 0.962/0.008/0.030 (133), m2 0.953/0.040/0.007 (149).

Differences (unpaired episode bootstrap, `pairwise.txt`): combined minus baseline: success +0.168 [+0.111,+0.225], fall -0.058 [-0.099,-0.018], timeout -0.110 [-0.154,-0.068]. Combined minus feedback alone: success +0.023 [-0.017,+0.062], fall -0.026 [-0.059,+0.005], timeout +0.003 [-0.020,+0.029]. Combined minus CRN alone: success +0.037 [-0.006,+0.081], fall +0.009 [-0.019,+0.037], timeout -0.046 [-0.082,-0.012].

## Success criterion (success >= 0.90, fall <= 0.04, timeout <= 0.06, no worse than either component)
- success 0.925: met. timeout 0.033: met. fall 0.043 (17 falls / 398): NOT met, by 0.003 (CI [0.025,0.065] contains 0.04). Verdict: criterion not strictly met; borderline on fall.
- "No worse than either component within noise": met (every difference above has a CI that includes zero or favours the combination). It is also not significantly better than the better component on any metric; median steps is the lowest of the four arms (43.5 vs 48.0 / 51.0 / 59.0).
- Compared with the true-simulator planner reference quoted in the brief (0.889/0.040/0.071, different protocol, N=100 M=12 H=12, not re-run here), the combined arm is numerically at or above it on success and timeout and similar on fall, with overlapping uncertainty.
- The baseline is lower here (0.757) than in d3's 300-step runs (0.827, seeds 0-2); gains over baseline in this table may be inflated by an unlucky baseline draw. The component-vs-component comparison is the more robust statement: the effects of feedback and CRN do not add up in a way this sample can resolve.

## Sample-efficiency sweep (`sweep_summary.txt`; eval seed 211; fb arms use step gains)
Model seeds 0,1 pooled (M=4 and M=8):
| arm | M=4 success/fall/timeout (eps) | M=8 success/fall/timeout (eps) |
|---|---|---|
| ol_indep | 0.726/0.201/0.073 (219) | 0.789/0.089/0.121 (190) |
| ol_crn3 | 0.831/0.130/0.039 (254) | 0.905/0.032/0.063 (221) |
| fbstep_indep | 0.851/0.079/0.070 (215) | 0.932/0.030/0.038 (234) |
| fbstep_crn3 | 0.895/0.091/0.014 (276) | 0.908/0.044/0.048 (249) |

M=16, model seed 0 only (about 100 episodes, CIs about +-0.08): ol_indep 0.777/0.053/0.170 (94), fbstep_crn3 0.934/0.033/0.033 (121); same seed at M=8: 0.806/0.102/0.092 and 0.845/0.086/0.069; at M=4: 0.697/0.229/0.073 and 0.909/0.068/0.023.
Reading: the combined arm at M=4 (0.895 pooled on m0,m1) beats the baseline at M=8 (0.789) and M=16 (m0: 0.777), i.e. the baseline does not reach the combined arm's M=4 success even with 4x the particles; the baseline is not monotone in M on one seed, so single-seed M=16 numbers are noisy. At M=4 the combined arm has the best success and timeout but a fall rate (0.091) above M=8 levels; fall is the metric that benefits least from CRN+feedback at low M. Only one evaluation seed and two model seeds: indicative, not conclusive.

## Compute (predictor rows per planning step, counted in the harness)
Open loop: E*N*M*H*iters = 122,880 rows at M=8. Feedback adds one nominal (u = 0) row per candidate per step, i.e. +1/M: 138,240 rows (+12.5%); wall-clock about +12-15% (586-604 s vs 511-528 s per 450-step run, under shared load). CRN adds no forward passes (it draws fewer random numbers). Combined at M=4 costs 76,800 rows against 245,760 for the baseline at M=16 (3.2x fewer).

## Diagnosis (offline, learned model, 64 start/goal pairs x 2 reps x 3 models; `diag.json`, `diag.log`)
Plan returned by each arm evaluated on fresh independent noise (256 particles) and on the planning noise:
| arm | cost, planning noise | cost, fresh noise | optimism (fresh - planning) | predicted fail prob |
|---|---|---|---|---|
| ol_indep | 2.006 | 1.999 | -0.006 | 0.055 |
| ol_crn3 | 1.901 | 2.036 | +0.135 | 0.065 |
| fbstep_indep | 1.884 | 1.901 | +0.016 | 0.046 |
| fbstep_crn3 | 1.820 | 1.901 | +0.081 | 0.044 |
(The indep rows have no retained planning noise; their "planning noise" column is a fresh M=8 draw, so optimism is about 0 by construction and omits candidate-selection bias.)
Observations: (i) the feedback arms have lower fresh-noise cost and predicted failure than open loop, consistent with Thm 4.5; (ii) with shared noise the returned plan is tuned to the 8 shared scenarios (optimism +0.08 to +0.14 in cost units) and its fresh-noise cost is not better than with independent noise (1.901 vs 1.901 for feedback; 2.036 vs 1.999 for open loop), so at start states CRN does not improve plan quality and the closed-loop gain from CRN must come from other effects (cross-replan consistency of the chained noise, or the lower variance of cost differences, Prop 3.2), which this single-replan diagnostic does not measure; (iii) feedback reduces the CRN optimism (0.135 to 0.081), plausibly because the gains absorb part of the scenario-specific tuning. These are hypotheses consistent with the numbers, not tested causes.
The fall miss is driven by model seed 0 (10 falls of 116, vs 2 of 110 for feedback with independent noise on the same seed); on seeds 1 and 2 the combined arm has 1/133 and 6/149 falls. Fall rates differ by model seed far more than between arms (fbstep_indep m2: 19/143), so the 0.043 vs 0.04 gap is within the noise of this protocol.

## Caveats
One evaluation seed; the 16 parallel environments of a run share a generator and episodes are bootstrapped as independent; arm comparisons are unpaired; sweep M=16 uses one model seed and two arms; the offline diagnostic uses start states only and the learned model, not the simulator. The d3/d6/true-simulator numbers quoted are from the brief or their own protocols (300 steps) and were not re-run here.
