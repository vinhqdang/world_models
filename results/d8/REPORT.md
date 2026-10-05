# D8: powered evaluation of the feedback-parameterised CEM planner

Benchmark cliff_hi, learned energy-score models `suite_hi/ckpt/es_s{0,1,2}.pt`, E=16, H=10, iters=3, M=8, 120-step episodes, streaming restart.
All numbers below come from `results/d8/summary_table.txt`, `oracle_summary.txt`, `derived.txt` (JSON per run beside them; per-episode records are stored in each JSON).

## Protocol
- Harness `experiments/d8_eval.py` (copy of the D3 planner path). Start/goal of the j-th episode of env slot i is drawn from a per-(seed, slot) generator and the wind signs from a per-seed generator, so arms run with the same seed see identical start/goal for each (slot, j). Planner randomness and trajectories differ, so pairing is partial.
- Eval seeds 100-102 (600 steps), 200-202 (300 steps), 300-302 (150 steps, open-loop top-up) per model seed 0-2. D3 used seeds 0-2 with a different generator layout, so the streams are disjoint.
- Arms: open loop N=32 (`ol32`); open loop N=36 (`ol36`, equal compute); feedback "step", kmax 7.5, N=32 (`fbs32`).
- Equal compute: feedback rolls (M+1)=9 particles per candidate (M sampled + 1 nominal) against M=8, so particle-steps per env step are 3*32*9*10=8640 against 7680 (+12.5%). Open loop at N=36 gives 8640, an exact match. Measured wall time per env step (shared CPU) was 1.17 s (ol32), 1.28 s (ol36), 1.33 s (fbs32).
- CIs: bootstrap over episodes, stratified by run (4000 resamples). Paired test: McNemar on matched (model, eval seed, slot, episode index). Unpaired: two-proportion z test.

## Main result (learned models, pooled over 3 model seeds)
| arm | episodes | success | fall | timeout | median steps to goal |
|---|---|---|---|---|---|
| open loop N=32 | 635 | 0.762 [0.729, 0.795] | 0.074 [0.055, 0.094] | 0.164 [0.135, 0.192] | 58.5 [55, 61] |
| open loop N=36 (equal compute) | 647 | 0.793 [0.760, 0.824] | 0.080 [0.059, 0.102] | 0.127 [0.100, 0.153] | 58.0 [55, 61] |
| feedback step N=32 | 694 | 0.890 [0.867, 0.914] | 0.055 [0.039, 0.072] | 0.055 [0.039, 0.072] | 48.0 [46, 51] |

Differences (feedback minus baseline):
- vs N=32: success +0.127 paired [+0.088, +0.167], McNemar p<1e-4 (n=545 matched episodes); timeout -0.116 [-0.149, -0.084]; fall -0.011 [-0.037, +0.015], p=0.50.
- vs N=36: success +0.103 paired [+0.065, +0.141], p<1e-4 (n=554); timeout -0.074 [-0.105, -0.043]; fall -0.029 [-0.058, -0.002], McNemar p=0.056 (unpaired p=0.061).
- Extra compute alone (N=36 vs 32): success +0.022 [-0.017, +0.061], p=0.30.
- Success is higher for feedback in each of the three model seeds (feedback / N=32 / N=36: 0.854/0.778/0.822, 0.908/0.708/0.797, 0.922/0.795/0.805). Median steps to goal drops by about 10 steps (Mann-Whitney p<1e-4). A length-bias-free subset (episodes started early enough to finish inside the window) gives the same picture (`summary_table.txt`, last block).

## Oracle (true simulator), E=100, 5 new seeds, 3.5k-5.3k episodes per arm
| arm | success | fall | timeout | fall/(fall+success) |
|---|---|---|---|---|
| open loop N=32 | 0.786 | 0.100 [0.090, 0.110] | 0.114 | 0.113 |
| open loop N=36 | 0.803 | 0.098 | 0.098 | 0.109 |
| feedback kmax 7.5 | 0.845 | 0.128 [0.119, 0.137] | 0.027 | 0.132 |
| feedback kmax 15 | 0.861 | 0.126 | 0.013 | 0.128 |

D3's oracle finding holds at power: falls rise by +0.028 against N=32 (p=8e-5; paired McNemar p=2e-3) and +0.030 against N=36 (p=2e-5), while timeouts fall by 0.087 and success rises by 0.059.
Decomposition (`derived.txt`): about 0.010 of the +0.028 is the mass of episodes that no longer time out (they are decided at the old fall hazard of 0.113); about 0.018 is a genuinely higher conditional hazard (0.113 to 0.132, p=0.014). So feedback makes the planner commit, and in addition it commits a little more riskily near the pit.

## Ablations (low power: 94-130 episodes each, eval seeds 200-202, exploratory)
| arm | episodes | success | fall | timeout | note |
|---|---|---|---|---|---|
| feedback + CVaR mix lam=0.5 | 130 | 0.854 | 0.069 | 0.077 | vs fbs32 on the same runs: success -0.062 paired (p=0.057), fall +0.015 (p=0.63), timeout +0.046 |
| feedback M=12 (+45% wall time) | 110 | 0.918 | 0.055 | 0.027 | vs fbs32: success -0.009 (p=1.0), no gain |
| feedback kmax 3.5 | 94 | 0.883 | 0.064 | 0.053 | vs fbs32: success -0.053 (p=0.27), no detectable difference |

On the learned models CVaR-mix scoring does not repair falls (there is no excess to repair) and costs some success through timeouts. On the oracle (5 seeds, `oracle_summary.txt`), CVaR-mix feedback gives fall 0.105, success 0.819, timeout 0.077 against 0.128 / 0.845 / 0.027 for plain feedback: the fall rate returns to the open-loop level (+0.005 vs N=32, p=0.52) and the success gain shrinks to +0.033. On the oracle, kmax 7.5 against 15 is indistinguishable (falls 0.128 vs 0.126). Open loop with CVaR is much worse (success 0.674, timeout 0.272).

## Analysis
1. The success gain survives at adequate power and is large: about +13 points over open loop at N=32 and +10 points over open loop at equal particle-steps, with CIs far from zero and the same sign in all three model seeds. The gain is not an N effect: N=36 buys +2 points (n.s.).
2. The gain comes mainly from timeouts (-12 and -7 points). Feedback agents reach the goal in about 10 fewer steps. Open-loop scoring of candidates is more pessimistic (the Thm 4.5 mechanism: open-loop spread accumulates over the horizon), so the open-loop planner hovers; closed-loop scoring prices the feedback-stabilised path correctly and commits.
3. Falls: on the learned models there is no increase. The point estimate is lower than open loop (-0.011 vs N=32, -0.029 vs N=36), with the equal-compute difference borderline (p=0.056). D3's single-run "falls down" (0.079 to 0.054) is thus in the right direction but not established. On the oracle the fall rate is higher by 2.8 points with a precise CI; that is the planner converting timeouts to decisions plus a modest hazard increase.
4. Oracle failure mode (mechanism is a hypothesis consistent with the numbers, not isolated by an experiment): at the optimum the planner scores a feedback law that is assumed to act with perfect state knowledge and unbounded replanning inside the horizon. This removes the risk-averse hovering of open loop, but the executed first action is the nominal one, and the stationary-gain closed loop still has a nonzero cumulative first-passage probability to the absorbing pit (the paper's item (ii) after Thm 4.5). The CVaR mix lowers the hazard back to the open-loop level at the price of timeouts, consistent with a trade between commitment and tail risk rather than a bug.

## Caveats
- Three learned models only; per-model success varies by up to about 0.09 within an arm (open loop N=32: 0.708 to 0.795), and episode-level bootstrap treats episodes as independent within a model. The consistent sign across models supports the success claim but 3 clusters cannot give a model-level CI.
- Streaming windows count completed episodes only; the start-cut subset removes that length bias and leaves conclusions unchanged.
- Open loop here is 0.762 vs D3's 0.827 for the same models (D3 had 191 episodes; different seeds); feedback is 0.890 vs 0.908. Sampling noise of the earlier single run explains this, so the gap is larger than D3 reported (+0.13 vs +0.08).
- Ablations are small and only the lam=0.5, M=12 and kmax=3.5 variants were run; "const" gains were not re-run (D3: 0.873 on 236 episodes). Ablation comparisons use the same runs and window of the reference arms.
- Pairing shares start/goal and wind-by-time-step only; discordant counts (e.g. 100/31 for success) show it helps modestly over unpaired tests, and the two give similar answers.
