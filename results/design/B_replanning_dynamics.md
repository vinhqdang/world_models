# Design B: closed loop of a sampling-based receding-horizon planner

## Verdict of the designer
No design under this stance reaches novelty probability 0.3; the best is 0.25, and the designer's own true-simulator pilot predicts it fails its kill test. The pilot indicates that the closed-loop pathologies behind the earlier StochNav results (hovering, plan flips) are largely artefacts of cold-started CEM and under-allocated particles, removed by known fixes.

## Pilot (true StochNav cliff_hi simulator, re-implementation; E=128, 360 steps, 700-1700 episodes per arm, ONE seed, N=32, H=10, 3 iterations; success / fall / timeout / flip rate)
Cold arms reproduce the earlier oracle numbers (open loop 0.770/0.130/0.100 against 0.786/0.100/0.114; feedback 0.864 against 0.845).
Open loop: independent cold 0.770/0.130/0.100/0.27; shared noise redrawn each replan cold 0.864/0.067/0.069/0.28; chained (refresh 0.2) cold 0.866/0.065/0.069/0.26; chained pure shift cold 0.838/0.092/0.070/0.26; independent warm 0.904/0.079/0.017/0.075; shared warm 0.903/0.080/0.017/0.10; chained warm 0.867/0.112/0.021/0.09; shared warm M=32 (4x compute) 0.872/0.028/0.101/0.08.
Feedback CEM: independent cold 0.864/0.107/0.029; shared cold 0.889/0.086/0.025; chained cold 0.884/0.100/0.016; shared warm 0.906/0.094/0.000; independent warm 0.907/0.092/0.001; chained warm 0.882/0.117/0.001; shared warm M=32 0.959/0.040/0.001; M=128 0.963/0.035/0.002; equal compute N=12, M=21 0.941/0.051/0.008; N=8, M=32 0.914/0.063/0.023; N=12, M=21 with persistent reference 0.959/0.036/0.005; damped plan averaging 0.697/0.302/0.001.
Readings (one seed, oracle only): chaining adds nothing over plain shared noise within a replan; warm start (missing from the default cem and fb_cem) removes timeouts and cuts flips threefold; with warm start, persisting scenarios across replans raises falls by 0.02-0.03; with feedback and warm start the residual error is particle noise; moving budget from candidates to particles at equal compute reaches 0.94-0.96, leaving at most about 2 points for any new planner at equal compute; damped averaging hurts because the wind step exceeds the control step.
On the cited 2018 precedent: the abstract of arXiv 1804.08766 does not mention reuse of fixed random-number sets across replans (full text not read; treat as unverified). The exact statistical precedent of the chaining formula is the correlated pseudo-marginal method (1511.04992).

## Candidate B1: Doob-ledger planning (novelty 0.25; estimated pass probability 0.10-0.15)
Carry across replans a finite scenario measure updated by conditioning on the abduced realised noise atom (two-atom noise map) and a contingent incumbent feedback policy; the sample-average value obeys a one-step martingale identity, and its departures (the refresh surprise R_t) gate plan switches (switch iff incumbent minus challenger exceeds the running mean of R).
Theorems: ledger identity V_{t+1} = V_t - c_t + D_t + R_t - I_t (pathwise); E[D_t | F_t] = 0 under the model when step c is conditioning; progress and switch-count bounds and a Ville inequality; newcomer-bias sketch. Nothing proved about the true environment under model misspecification.
Prior art: 1005.3605, Lucia et al. 2013, 2204.09607, 2411.15651, 2610.06002 (P3, persists a population of plans across replans, closest 2026 paper), 1606.06056, 1307.5640, 1206.0038, Scokaert-Mayne-Rawlings 1999, 2607.06945, 2607.04006, 1507.02780, 1902.08967, 2008.06389, 1301.3878, 1511.04992, 2506.23097, 2609.13845, MCTS subtree reuse (POMCP, Stochastic MuZero). The conditioning step is the sampled-fan version of scenario-subtree rolling and MCTS subtree reuse.
Kill test: Stage 0 replicates the pilot on the learned models; stop if warm start, the N=12/M=21 reallocation and the persistent reference reach 0.93 with timeouts at most 0.01. Stage 1: 11 arms at equal compute, 600 episodes per arm; pass requires B1 success at least best of known-fix arms plus 0.03 with interval excluding 0, fall not higher by more than 0.01, flip rate at most 0.7 of the warm-start arm, and the oracle martingale check.

## Candidate B2: continuation-closed averaging (novelty 0.15; pilot evidence against)
Average plan parameters across replans along the dynamic-programming continuation map with a Kalman-like gain; consistent only if the class is continuation-closed. Reduces largely to tube MPC plus DMD-MPC; damping lost 0.08-0.21 success in the pilot.

## Candidate B3: paced particle budget across replans (novelty 0.15)
Neyman allocation of particles across replans by the paired-difference standard deviation among elites, with pacing; cousin of racing (dead list); a crude trigger did not beat constant M=21.

## Recommendation of the designer
Do not commit to a new-algorithm paper from this stance. Run Stage 0 first on the learned models. If the known fixes reach the oracle levels, the chained-noise and hovering narrative must be rewritten as a cold-start effect and this stance has no headroom on StochNav.

Pilot scripts were in the session scratchpad (b/pilot.py, b/pilot_fb.py, b/pilot_alloc.py), not in the repository.
