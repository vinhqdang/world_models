"""Writes results/paper/PROTOCOL.md from the cell definitions (cwk/cells.py), the calibration (results/cwk/CALIBRATION.json) and the
text below. Run after calibration, before any R1-R6 run; the protocol is committed before data of the sealed cell are drawn."""
import sys, json, os
sys.path.insert(0, "/home/user/world_models")
sys.path.insert(0, "/home/user/world_models/experiments")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells

OUT = "/home/user/world_models/results/paper/PROTOCOL.md"
CAL = json.load(open("/home/user/world_models/results/cwk/CALIBRATION.json"))
COMPUTE = open("/home/user/world_models/results/paper/compute_estimate.md").read() if os.path.exists("/home/user/world_models/results/paper/compute_estimate.md") else "(see below)"


def chdesc(w):
    parts = []
    for c in w.channels:
        k = c["kind"] + (f"[axis {c['axis']}]" if "axis" in c else "")
        parts.append(f"{k} {c['scale']:.4f}")
    for f in w.forces:
        parts.append(f"force({f['body']}) {f['scale']:.4g}")
    return "; ".join(parts)


def lawdesc(w):
    l = w.laws
    if l is None:
        return "tanh (authors)"
    return {"tanh": "tanh", "gated": f"gated pi={l.get('pi')}", "gauss_clip": f"gauss_clip c={l.get('c')}"}[l["kind"]]


def table(names):
    rows = ["| cell | law | iid fraction | drivers r (channels / forces) | action gain | seed offset |", "|---|---|---|---|---|---|"]
    for n in names:
        w = cells.ALL_PAPER[n]()
        g = w.overrides.get("action_gain") if w.overrides else None
        rows.append(f"| {n} | {lawdesc(w)} | {w.iid_fraction} | {len(w.channels) + len(w.forces)}: {chdesc(w)} | {g:.4f} | {w.seed_offset} |" if g else
                    f"| {n} | {lawdesc(w)} | {w.iid_fraction} | {len(w.channels) + len(w.forces)}: {chdesc(w)} | (authors) | {w.seed_offset} |")
    return "\n".join(rows)


def cal_table():
    rows = ["| key | sigma | gain | dev oracle % | verification oracle % | within 50 +- 8 | note |", "|---|---|---|---|---|---|---|"]
    for k, v in CAL.items():
        if k.startswith(("info_", "misalign_")):
            rows.append(f"| {k} | {v['sigma']} | {v['gain']:.4f} | {100 * v['dev_oracle']:.1f} | {100 * v['verify']['oracle']:.1f} | "
                        f"{'yes' if v['in_window'] else 'NO'} | {'refined (second round)' if v.get('refined') else ''}{' at search bound' if v.get('capped') else ''} |")
        elif k.startswith("force_unit"):
            rows.append(f"| {k} | sigma_ref {v['sigma_ref']} | force {v['force']:.4g} | - | - | - | unit = force / sigma_ref = {v['unit']:.4g} |")
    return "\n".join(rows)


envs = list(cells.INFO_ENV)
info_names = [f"{e}_{c}" for e in envs for c in ("P", "T", "G", "pi5", "pi10", "pi50", "pi100", "rd", "rd2", "Piid50", "Pdata1", "Pdata2")]
stage2 = {k: v for k, v in CAL.items() if k in ("e1_nu50_gain", "e1_nu80_gain", "e1_nu50_sat", "e1_nu50_deadzone", "e2", "e3")}

TEXT = f"""# Protocol of the analysis-paper experiments (written before any R1-R6 run)

Plan: `paper_cf/PLAN.md` (sections 2 and 3). Code: `cwk/` (sim.py, worlds.py, cells.py, eb.py, pipeline.py, oracle.py, runner.py),
`experiments/paper_*.py`. Phase 1 (code, R0 unit tests, calibration, this protocol) is complete; no result of R1-R6 exists when this
file is written. The sealed cell is drawn after this file is committed (section 9). Nothing below is changed after results are seen;
every later deviation is recorded in a section "Deviations" appended to the corresponding R*.md.

## 1. Setting

State-level same-disturbance counterfactual transport on the authors' simulators (physics untouched, upstream commit 5c84bba, clone
`/home/user/cwm_upstream`, unmodified), protocol of Stage 2 (`results/cwk/STAGE2.md`): episodes of 5 transitions (t = 0, 1 behaviour policy;
t = 2 anchor; t = 3, 4 uniform actions), 16384 factual episodes per cell; evaluation population 512 parents x 64 draws x 3 query actions
(plus the executed action), counterfactual R^2 on the primary coordinate group (wrist d = 2, cup position d = 2, gripper position d = 3);
development population 64 x 32 (disjoint reset seeds, used only by privileged reference arms). Mean predictor mu_hat(H, a): MLP 3 x 384
(SiLU) on the last three states and the action, 20 epochs, batch 512, 2-fold cross-fitting by episode (residuals), a model on all data for queries.
Kernel / EB arms work on the residual of the scored coordinates in standardised units; psi(a) = (1, a_axes / a_lim) (declared actuated axes).

## 2. Cells

### 2.1 Reused Stage 2 cells (cached data in results/cwk/data; amplitudes from Stage 2 calibration, unchanged)

R1 and R6 use the nine cells with cached data: E1_nu50_gain_iid50, E1_nu80_gain_iid50, E2_cup_iid50, E3_fetch_iid50, E1_nu50_sat_iid50,
E1_nu50_deadzone_iid50 (iid fraction 0.5) and E1_nu50_gain_iid100, E2_cup_iid100, E3_fetch_iid100 (iid fraction 1). Amplitudes (calibrated in Stage 2):

```json
{json.dumps({k: {a: b for a, b in v.items() if a in ('sigma', 'gain', 'wind', 'load')} for k, v in stage2.items()})}
```

### 2.2 Information-limited cells INFOLIM(env, law, pi, r) (new)

Environments E1o = HandReach (d = 2, wrist; upstream config hand_affine100 with the declared axes 0, 1), E2o = Ball-in-Cup (d = 2, cup position),
E3o = FetchReach (d = 3, gripper position; goal-position command). Drivers (independent, all with the same law): r = d gains, one per
actuator (command perturbation s eps x_j on actuator j only: channel kind `gain`, key `axis` of cwk/sim.py); r = d + 1 adds a common offset
(direction = the authors' alternating +-1 on the declared axes); r = d + 2 adds a force on a body (hand: palm, vector (9, 0, 30); cup: ball, +x;
fetch: gripper link, +x). Per-axis command budget sigma: gain channels sigma (1 - 1/2), offset sigma / 2 (r = d: gains sigma); command gain
g = 0.98 (1 - sigma) / a_lim, so the disturbed command never leaves the native range (checked by R0). Force scale = force_unit x sigma.
Driver laws (cwk/sim.py `law_eps`): `tanh` eps = tanh(driver) (default); `gated(pi)` eps = gate x (2 Phi(driver) - 1) with gate ~ Bern(pi)
from a separate deterministic stream per (seed, parent, draw, driver), shared by all counterfactual branches of a draw (R0 verifies:
prefix stability, rate, independence, replay of every branch with the stored eps); `gauss_clip` eps = clip(driver / 3, -1, 1). For
iid fraction 1 the driver is N(0, 1), so the gated law is Bern(pi) x U(-1, 1). The driver of the persistent cells (Piid50) mixes the
authors' episode field with innovations (iid fraction 0.5), the gate being independent across transitions.

Cells (R2a; names `<env>_<cell>`):

* P: gated pi = 0.2, r = d + 1; T: tanh, r = d + 1; G: gauss_clip, r = d + 1;
* pi5, pi10, pi50, pi100: gated pi = 0.05, 0.1, 0.5, 1 at r = d + 1; rd, rd2: gated pi = 0.2 at r = d and r = d + 2;
* Piid50: P at iid fraction 0.5; Pdata1, Pdata2: two further data realisations of P (reset and driver seeds differ by 1e6, 2e6);
* SEALED: one cell drawn after the protocol commit (section 9).

{table(info_names)}

Calibrated amplitudes (outcome-blind oracle R^2 about 50 +- 8 on development cohorts; method of `experiments/paper_calibrate.py`,
details and scans in `results/paper/CALIBRATION.md`, `results/cwk/CALIBRATION.json`). The cells share a key where they share an
amplitude (Pdata1, Pdata2 use the P key; Piid50 has its own key calibrated at iid fraction 0.5):

{cal_table()}

### 2.3 MISALIGN (R5)

Ball-in-Cup, tanh law, iid fraction 1: MIS_E2o = thrust-misalignment driver (command perturbation s eps J x on the two actuated axes,
J = [[0, -1], [1, 0]], kind `rotate`) plus a common-offset driver (r = d = 2); MIS_E2o_nooff = the rotate driver alone (r = 1).

{table(['MIS_E2o', 'MIS_E2o_nooff'])}

### 2.4 Upstream cells (R3)

The authors' own configurations, single offset driver (hand_l0 = their `hand`, cup, fetch), iid fraction 0.5 variants (hand_iid50, cup_iid50, fetch_iid50),
hand_load (body force), hand_affine50, hand_affine100 (gain fractions 0.5 and 1). Amplitudes, gains and population seeds are the authors' (no calibration).

### 2.5 Synthetic worlds

R4 (d = 1) and R5 (d = 2) worlds are defined in `experiments/paper_r4.py`, `experiments/paper_r5.py` (200k factual, 40k test, noise sd 0.05, uniform
drivers; seeds fixed in the scripts). The K0 generator (syn2, `cwk/eb.py: syn2_generate`) is used only by R0.

## 3. Arms (counterfactual R^2 on the primary group; * = privileged simulator information)

Every fit seed refits the mean predictor, the HH / EB / head models on 90% of the transitions (episode-wise split, seed-determined); the held-out 10%
is used only for selection (nugget, K, restarts, lambda, checkpoints). 5 fit seeds (3 for sweep and extra-data cells of R2b).

* trained, rr_learned (residual reuse with the learned mean), rr_exact* (exact means), oracle* (outcome-blind), copy*;
* hh_global, hh_binned (Stage 2 definitions: moment estimator on psi_j psi_k, Burer-Monteiro lift rank 4, ridge 1e-3, 1 or 8 k-means context bins);
* hh_binned_nug: hh_binned with a nugget N_i = c tr(C_i) / d, C_i = G_i G_i^T, c in {{0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}} (c = 0 means 1e-6) chosen by the held-out
  factual Gaussian log-likelihood; transport G' (G^T G + N I)^-1 G^T r; hh_nug_ref*: c chosen on the development counterfactuals (privileged);
* hh_basis_{{x, abs, x_x3, cubic}}: one bin, psi = (1, x), (1, |x|), (1, x, x^3), (1, x, x^2, x^3) per actuated axis, nugget as hh_binned_nug, diagnostic cond(Phi^T Phi);
* rc_eb_gmm: HH lift (one bin, rank R = number of drivers of the cell, at least 2) + full-covariance Gaussian-mixture prior on xi by marginal likelihood
  (K in {{16, 32}}, 2 restarts each = best of 4 by held-out NLL, 3000 steps, Adam 1e-2 cosine, batch 1024), posterior mean in closed form;
* rc_npmle: same lift, 2000 grid atoms (regular grid for R <= 3, scrambled Sobol otherwise) on [-b, b]^R (b = 2 x the 99.5% quantile of the min-norm coefficients, in [3, 8]),
  EM 150 iterations on 20000 factual points, noise variance c tr(C)/d selected by held-out NLL over c in {{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}};
* rc_flow: same lift, RealNVP prior (6 affine couplings), trained by the importance-sampled marginal likelihood (1024 shared prior samples per step, 1500 steps),
  posterior mean by self-normalised importance sampling with 4096 prior samples;
* idt: independent-factor model (K = 4 location-mixture sources per driver, learned columns initialised at the HH lift, 3000 steps, 4 restarts for R <= 3 and 1 restart for R >= 4 because of the 4^R enumeration cost, held-out NLL);
  glr_iso: best-subset GLR isolation with the idt columns, lambda in {{2, 4, 6, 8, 12}} by the held-out factual NLL of the subset mixture; glr_iso_lamo*: lambda by the counterfactual loss (privileged);
* ls_signed_nug, cvae, pck_pol: as in Stage 2 (`results/cwk/STAGE2.md`);
* oracle_kernel* (exact mean, parent conditional covariance, two folds over draws), oracle_kernel_lm* (learned mean, parent map), oracle_quad* (learned mean, per-parent quadratic regression
  on the executed outcome; oracle_quad_ex*: exact mean, all pairwise products), oracle_struct_bayes* (per-parent finite-difference Jacobian (h = 0.5) in the drivers, exact
  Bayes over the world's driver laws by pattern enumeration and quadrature on the linearised map, likelihood noise = linearisation misfit; reports the linearisation R^2).

## 4. Seeds and sizes

Fit seeds 0-4 (R2b: 0-4 for P, T, G, Piid50, SEALED; 0-2 for pi5, pi10, pi50, pi100, rd, rd2, Pdata1, Pdata2). Data: 16384 episodes, 512 parents x 64 draws, 64 x 32 development.
R4 synthetic seeds 0-4, R5 synthetic seeds 0-2.

## 5. Metrics

Pooled counterfactual R^2 = 1 - mean loss / target variance over parents x draws x 3 queries (authors' definition); per-parent losses of every arm are saved
(`results/paper/raw/*.json`), per (parent, action) losses for R4. Held-out factual NLL (standardised units) for the HH / EB arms; cond(Phi^T Phi) for basis variants;
realised oracle R^2 and rho of every evaluation cell (calibration check).

## 6. Statistics

Whole-parent x fit-seed paired bootstrap, 2000 resamples (fixed seed 2026), 95% percentile intervals; paired differences use the same resamples; Holm across environments
where several environments test one hypothesis. Predictions are evaluated on point estimates with the rules in section 7 and reported with their intervals.

## 7. Pre-registered predictions (verbatim from the plan)

* P1 saturation: on Stage 2 cells, hh_binned_nug, rc_eb_gmm, oracle_quad* within 1.0 of oracle_kernel_lm*.
* P2 Gaussian control: |rc_eb_gmm - hh_binned_nug| <= 1.0 on G cells.
* P3 headroom: rc_eb_gmm - hh_binned_nug increases as pi decreases and as r - d increases; >= 0 at pi = 1 within CI.
* P4 sign: basis (1,|x|) matches (1,x) on held-out factual NLL within 0.01 nats and loses >= 10 points on cross-zero queries in hand_affine100.
* P5 no harm: on upstream additive cells every HH/EB arm within 1.5 of rr_learned.
* P6 rotation: HH lift below RR in the rotation world; within 1 of oracle in the misalignment world.

Evaluation rules as implemented in `experiments/paper_report.py` (fixed here): P1: |difference of pooled R^2| <= 1.0 for the three arms on each of the nine cells (verdict: all nine);
oracle_quad* uses the learned mean so that it is comparable with oracle_kernel_lm*. P2: the three G cells. P3: per environment, the sequence over pi in (0.05, 0.1, 0.2, 0.5, 1) is
non-increasing and over r - d in (0, 1, 2) non-decreasing in the point estimates, and the upper confidence limit at pi = 1 is >= 0. P4: held-out NLL difference (abs minus x) <= 0.01 nats in
absolute value and cross-zero R^2 loss of (1,|x|) relative to (1,x) >= 10 points, on hand_affine100 (cross-zero: some actuated axis changes sign between executed and query action).
P5: |arm - rr_learned| <= 1.5 for hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, rc_npmle, rc_flow, pck_pol on every upstream cell. P6: synthetic worlds of R5, rotation (I + x_1 J): hh_lift (R = 2)
below rr; misalignment + offset: |hh_lift (R = 3) - oracle| <= 1. The paper is written whatever the outcome.

## 8. Runs and order

R0 unit tests (done; `results/paper/R0.log`, `R0.md`: old cells bit-identical to the cached Stage 2 generator, gates shared across branches, rc_eb_gmm / idt / glr_iso / quadrature reproduce K0 exactly on g15, g30, tanh seed 0) -> R1 (`experiments/paper_r1.py`) || R2a (`paper_r2.py gen`) -> R2b (`paper_r2.py ladder`) -> R3 (`paper_r3.py`) -> R4, R5, R6
(`paper_r4.py synth|real`, `paper_r5.py synth|real`, `paper_r6.py`) -> R2c (`paper_r2.py validate E1o_P`, 64 parents); reports `paper_report.py r1 ... r6`.
Outputs `results/paper/R*.json|md`, raw per-seed files `results/paper/raw/`, logs `results/paper/logs/`.

R1: all nine cells, arms rr_learned, trained, hh_global, hh_binned, hh_binned_nug, hh_nug_ref*, oracle_kernel_lm*, oracle_quad*, rc_eb_gmm, pck_pol (+ basis variants on E1_nu50_sat_iid50).
R2b ladder arms: ls_signed_nug, hh_binned, hh_binned_nug, hh_nug_ref*, glr_iso, glr_iso_lamo*, rc_eb_gmm, rc_npmle (P, T, G, Piid50, SEALED only), rc_flow, cvae, idt, oracle_kernel_lm*, oracle_quad* (+ trained, rr_learned and the deterministic
oracle*, rr_exact*, oracle_kernel*, oracle_quad_ex*, oracle_struct_bayes*). R3: hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, rc_npmle, rc_flow, pck_pol, oracle_kernel_lm*. R4 real: hh_basis_x, hh_basis_abs,
oracle_kernel_lm* on hand_affine100 and E1_nu50_gain_iid100. R5 real: hh_global, hh_binned_nug, rc_eb_gmm, oracle_kernel_lm* on MIS_E2o, MIS_E2o_nooff. R6: iid100 cells, mean predictor trained on the first 512 episodes
(cross-fitted inside them, out-of-sample elsewhere), arms hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, oracle_kernel_lm*.
R2c: full-simulation quadrature (grid 11 per active driver, kernel width half a grid cell) vs oracle_struct_bayes* on E1o_P, 64 parents.

### Compute estimate

{COMPUTE}

## 9. Sealed cell

After this protocol is committed, `python3 experiments/paper_sealed.py <protocol commit hash>` draws the environment (uniform over E1o, E2o, E3o), pi (log-uniform on [0.08, 0.6])
and an amplitude factor (U[0.85, 1.15] applied to the calibrated sigma interpolated in log pi over the calibrated gated cells) from an RNG seeded with the first 16 hexadecimal
digits of the hash, and writes `results/paper/sealed_cell.json` (hash, seed, every draw). The cell has r = d + 1, iid fraction 1 and seed offset 3000000. The hash is committed with the
file before any data of the cell are generated; no recalibration or inspection of its outcomes is allowed; its realised oracle R^2 is reported as it comes.

## 10. Deviations from the plan (disclosed before any run)

See the final report of phase 1 (listed also in `results/paper/CALIBRATION.md`): (1) rc_eb_gmm uses 2 restarts per K (best of 4 overall) instead of 4 per K, to keep the cost of R1 and R2b at
about half; the K0 reproduction in R0 uses the K0 setting (K = 32, 4 restarts). (2) oracle_quad* is defined with the learned mean (oracle_quad_ex* is the exact-mean variant of diag4.py).
(3) Gaussian control = gauss_clip with c = 3 (clipping probability 0.27% per driver). (4) glr_iso, glr_iso_lamo* and idt need the independent-factor columns and therefore the idt fit.
(5) idt restarts: 4 for R <= 3, 1 for R >= 4 (cost). (6) Calibration: keys whose verification cohort fell outside 50 +- 8 in the first round (E1o pi5; E2o pi5, pi10, rd; E3o pi5, pi10) were re-searched once on the verification cohort and verified on a third cohort (marked refined in the table; all end within the window). (7) rc_npmle is run in R2b only on the P, T, G, Piid50 and SEALED cells (not on the sweep and extra-data cells); R3 keeps 5 seeds. (8) Development cohorts are small (40 x 24), so the oracle R^2 of an evaluation population differs from the calibration target by sampling error of a few points; the realised value is reported. (13) R = number of drivers is given to the EB/idt lift (privileged structural information, applied identically to all EB arms); this makes explicit what item 5 of the lift rank already stated.
"""
open(OUT, "w").write(TEXT)
print("written", OUT, len(TEXT))
