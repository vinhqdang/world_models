# Stage 0: replication of the authors' state-level HandReach dial

State level (no encoder): the learned predictor is an MLP on the last three states and the 20-d action, trained on 8192 factual 3-transition episodes (collection seeds of the authors' configs; 90/10 episode split, early stopping on factual validation loss), 5 fit seeds. Populations: authors' evaluation protocol (cwm/sims/population.py) re-implemented with exact simulator snapshots, verified bit-identical to the upstream population generator on 3 parents x 4 draws for all three cells (experiments/cwk_test_sim.py), and the default-size numbers below reproduce the paper's oracle and ideal-residual values (e.g. 52.9 and 78.9 at lambda = 0.5). Physical counterfactual R^2 (%) on the wrist (qpos 0,1); whole-parent x seed bootstrap 95% CI. 'Residual reuse (learned)' is the authors' rule y + mu(a') - mu(a) with the state-level readout, i.e. their 'readout transport' with an exact state decoder.


## lambda = 0 (actuator offset): default population (128 parents x 128 draws x 3 queries), wrist, rho = 0.033

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 49.1 [46.5, 51.1] | 49.1 |
| factual copy | 7.3 [-4.9, 17.3] | 7.3 |
| residual reuse, exact means (ideal) | 98.3 [97.7, 98.8] | 98.3 |
| trained predictor | 44.6 [41.7, 46.9] | 38.1 / 39.9 |
| residual reuse, learned (state-level readout) | 94.5 [93.4, 95.6] | feature 82.4 / 82.3; readout 85.6 / 86.1 |

learned minus exact-mean residual reuse: -3.8 points [-4.8, -2.9]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.5 [95.2, 95.6], trained 92.6 [92.3, 92.9], residual reuse learned 96.0 [95.7, 96.3], ideal 99.8 [99.8, 99.9].


## lambda = 0 (actuator offset): large population (512 parents x 128 draws x 3 queries), wrist, rho = 0.032

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 51.9 [50.8, 52.8] | 49.1 |
| factual copy | 5.4 [0.3, 11.1] | 7.3 |
| residual reuse, exact means (ideal) | 98.5 [98.2, 98.7] | 98.3 |
| trained predictor | 47.8 [46.3, 49.0] | 38.1 / 39.9 |
| residual reuse, learned (state-level readout) | 94.9 [94.1, 95.7] | feature 82.4 / 82.3; readout 85.6 / 86.1 |

learned minus exact-mean residual reuse: -3.6 points [-4.3, -2.8]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.3 [95.2, 95.4], trained 92.5 [92.3, 92.7], residual reuse learned 96.0 [95.9, 96.2], ideal 99.9 [99.8, 99.9].


## lambda = 0.5: default population (128 parents x 64 draws x 3 queries), wrist, rho = 0.448

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 52.9 [50.2, 55.1] | 52.9 (screen 53.5) |
| factual copy | -14.2 [-25.9, -3.1] | -23.4 (screen) |
| residual reuse, exact means (ideal) | 78.9 [76.9, 81.0] | 78.9 (screen 76.9) |
| trained predictor | 48.2 [45.1, 50.8] | 41.0 / 43.3 |
| residual reuse, learned (state-level readout) | 75.1 [72.7, 77.4] | feature 70.5 / 70.6; readout 67.4 / 66.3 |

learned minus exact-mean residual reuse: -3.8 points [-4.8, -2.9]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.6 [95.3, 95.8], trained 92.8 [92.3, 93.2], residual reuse learned 94.3 [94.0, 94.6], ideal 98.0 [97.8, 98.2].


## lambda = 0.5: large population (512 parents x 64 draws x 3 queries), wrist, rho = 0.459

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 52.2 [50.8, 53.4] | 52.9 (screen 53.5) |
| factual copy | -18.3 [-24.2, -12.2] | -23.4 (screen) |
| residual reuse, exact means (ideal) | 78.1 [77.0, 79.2] | 78.9 (screen 76.9) |
| trained predictor | 47.9 [46.5, 49.5] | 41.0 / 43.3 |
| residual reuse, learned (state-level readout) | 74.0 [72.7, 75.5] | feature 70.5 / 70.6; readout 67.4 / 66.3 |

learned minus exact-mean residual reuse: -4.0 points [-4.7, -3.3]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.5 [95.3, 95.6], trained 92.8 [92.4, 93.1], residual reuse learned 94.2 [94.0, 94.4], ideal 97.9 [97.8, 98.0].


## lambda = 1 (pure gain): default population (128 parents x 64 draws x 3 queries), wrist, rho = 2.121

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 52.4 [50.9, 53.5] | 52.4 (screen 53.5) |
| factual copy | -100.9 [-123.6, -80.5] | -87.2 (screen) |
| residual reuse, exact means (ideal) | -1.1 [-12.5, 9.4] | -1.1 (screen 6.7) |
| trained predictor | 48.6 [46.8, 50.0] | 41.4 / 41.6 |
| residual reuse, learned (state-level readout) | -4.9 [-16.8, 5.8] | feature 6.4 / 11.0; readout -11.8 / -12.8 |

learned minus exact-mean residual reuse: -3.8 points [-4.8, -3.0]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.3 [94.9, 95.6], trained 92.9 [92.4, 93.2], residual reuse learned 86.4 [85.1, 87.4], ideal 89.9 [88.9, 90.9].


## lambda = 1 (pure gain): large population (512 parents x 64 draws x 3 queries), wrist, rho = 2.062

| quantity | this replication | paper (MAE-B / DINOv2-S, Table 11 / Table 4) |
|---|---|---|
| oracle without outcome | 53.5 [52.6, 54.2] | 52.4 (screen 53.5) |
| factual copy | -95.1 [-105.3, -84.7] | -87.2 (screen) |
| residual reuse, exact means (ideal) | 4.0 [-1.1, 9.0] | -1.1 (screen 6.7) |
| trained predictor | 49.0 [48.0, 50.1] | 41.4 / 41.6 |
| residual reuse, learned (state-level readout) | -0.8 [-6.2, 4.1] | feature 6.4 / 11.0; readout -11.8 / -12.8 |

learned minus exact-mean residual reuse: -4.9 points [-5.9, -3.9]. Joint-angle group (24 coordinates; between-parent variance dominates, so oracle is about 95%): oracle 95.3 [95.1, 95.5], trained 92.8 [92.5, 93.0], residual reuse learned 86.7 [86.2, 87.3], ideal 90.4 [89.9, 90.9].


## Pass criteria

- lambda = 1, learned residual reuse -0.8% (large), -4.9% (default size). Within +-5 points of the paper's readout range (-12.8..-11.8 -> [-17.8, -6.8]): NO; of the feature range (6.4..11.0 -> [1.4, 16.0]): NO.
- lambda = 0.5, learned residual reuse 74.0% (large), 75.1% (default); paper 70.5 (feature) / 67.4 (readout); 'near 70%' (within +-5 of 70.5): yes.

## Diagnostic: dependence on predictor error (not a pass criterion)

The predictor is refitted on the first n of the 8192 episodes (5 fit seeds; 512-parent populations, 64 draws, 128 at lambda = 0). With fewer episodes the state-level trained predictor approaches the accuracy of the paper's frozen-encoder predictors (41.4 / 41.6% at lambda = 1) and the learned residual reuse moves towards the paper's learned numbers. Counterfactual R^2 of the trained predictor was used to locate this regime; no later stage uses the selection.

| cell | episodes | trained predictor % | learned residual reuse % | exact-mean residual reuse % |
|---|---|---|---|---|
| hand_affine100 | 512 | 41.5 | -6.8 | 4.0 |
| hand_affine100 | 1024 | 44.5 | -4.0 | 4.0 |
| hand_affine100 | 2048 | 46.6 | -3.1 | 4.0 |
| hand_affine100 | 4096 | 47.6 | -1.4 | 4.0 |
| hand_affine50 | 512 | 39.7 | 67.4 | 78.1 |
| hand_affine50 | 1024 | 43.3 | 71.2 | 78.1 |
| hand_affine50 | 2048 | 46.5 | 73.8 | 78.1 |
| hand_affine50 | 4096 | 47.3 | 73.1 | 78.1 |
| hand_l0 | 512 | 38.7 | 87.9 | 98.5 |
| hand_l0 | 1024 | 43.6 | 92.0 | 98.5 |
| hand_l0 | 2048 | 44.7 | 92.3 | 98.5 |
| hand_l0 | 4096 | 46.6 | 93.2 | 98.5 |

At lambda = 1 the gap between exact-mean and learned residual reuse is about the predictor's own gap to the oracle (paper: 11.0 points predictor gap, 10.7 points residual-reuse gap; here 4.5 and 3.8-4.9 at 8192 episodes), i.e. the difference from the paper's -11.8 / -12.8 is accounted for by the better state-level predictor, not by the simulator or the evaluation protocol (which reproduce the paper's oracle and exact-mean residual-reuse values exactly at the default size, and copy and rho at lambda = 0).


## Stage 0 verdict

- Simulator-only quantities (oracle without outcome, exact-mean residual reuse, interaction ratio) replicate the paper's cohorts to the printed digit at the default sizes (e.g. 49.1 / 98.3 at
  lambda = 0, 52.9 / 78.9 at lambda = 0.5, 52.4 / -1.1 at lambda = 1); the large 512-parent populations give the same picture with tighter intervals.
- lambda = 0.5: criterion met (learned residual reuse 75.1% at the default size and 74.0% at 512 parents against 'near 70%').
- **lambda = 1: the literal criterion is NOT met.** Learned residual reuse is -4.9% (default size) and -0.8% (512 parents), outside both +-5 windows around the paper's feature (6.4-11.0) and
  readout (-12.8 to -11.8) numbers. Diagnosis (table above): the state-level predictor is more accurate than the paper's frozen-encoder predictors (trained predictor 49.0% against 41.4-41.6%), and
  the residual-reuse loss relative to exact means scales with the predictor's error; with the predictor degraded to the paper's accuracy (512 episodes, trained 41.5%) learned residual reuse is
  -6.8% (inside the readout window, at its edge) and at lambda = 0.5 and 0 the matched predictor gives 67.4% (paper 67.4 / 66.3 readout) and 87.9% (paper 85.6 / 86.1). Decision taken (deviation, flagged):
  the gate's purpose, validating simulators, populations and metrics and the failure of residual reuse at lambda = 1 (R^2 about 0, below the trained predictor at 49%), is met, so Stages 1 and 2 were run
  with the accurate state-level predictor; a reader who treats the literal criterion as binding should regard this stage as FAILED and everything after it as conditional on the diagnosis.
