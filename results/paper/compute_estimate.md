Measured on this machine (4 cores, 2 torch threads per heavy job; timings in `results/paper/logs/timing_*.json`, simulator timings from `cwk/worlds.py`):
one GMM fit (3000 steps) K = 32 / K = 16: 112 s / 58 s (d = 2, R = 3), 173 s / 84 s (d = 3, R = 4); RealNVP fit 36-43 s; NPMLE (5 noise levels x 150 EM iterations) 260-290 s; one idt fit (3000 steps, 4^R combinations) 128 s (R = 3), 644 s (R = 4), about 2600 s expected for R = 5 (hence idt restarts 4 for R <= 3 and 1 for R >= 4); mean predictor (2-fold + full) about 1.5-2 min; HH binned fits about 1 min per model.
Simulation per cell (E1o): factual collection 7 min, evaluation population with finite-difference Jacobians 10 min, development population 1.5 min (about 19 min per cell; E2o and E3o similar).

| item | seed-runs | minutes per seed-run | job-hours |
|---|---|---|---|
| R2a data (36 cells + sealed) and R3 data (cup, fetch and iid50 variants) | - | 19 per cell | about 13 simulator-hours |
| R1 (9 cells) | 45 | about 12 | 9 |
| R2b E1o and E2o (44 seed-runs each; rc_npmle only on P, T, G, Piid50) | 88 | about 22 | 33 |
| R2b E3o (R = 4, 5; idt 1 restart) | 44 | about 30 (rd2: 65) | 23 |
| R2b sealed cell | 5 | about 30 | 2.5 |
| R3 (9 upstream cells) | 45 | about 17 | 12.5 |
| R4 (synthetic 8 worlds x 5 seeds, 2 real cells x 5 seeds) | - | - | 2.3 |
| R5 (synthetic 3 seeds, 2 MISALIGN cells x 5 seeds) | - | - | 2.8 |
| R6 (3 cells x 5 seeds) | 15 | about 11 | 2.8 |
| R2c (64 parents) | - | - | 0.6 |
| **total** | | | **about 89 heavy job-hours plus about 13 simulator-hours** |

With two heavy jobs and one simulator job at a time this is about 48 hours of wall-clock time (the plan assumed about 45 job-hours). Levers applied: idt 1 restart for R >= 4; rc_npmle only on P, T, G, Piid50, SEALED in R2b; R3 stays at 5 seeds.
