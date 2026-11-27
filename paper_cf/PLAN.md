# Plan: analysis paper on same-disturbance counterfactual transport

Target journal: Neural Networks (Elsevier, elsarticle). Backups: International Journal of Approximate Reasoning; Machine Learning (Springer, sn-jnl).
Title: "When Can Counterfactual World Models Beat Residual Reuse? Saturation, Gauge, and Empirical-Bayes Limits of Same-Disturbance Transport".
Author: Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam, vinh.dq4@buv.edu.vn, ORCID 0000-0002-3877-8024.
Paper type: analysis (theory plus benchmark). No new algorithm is claimed. State level only; no pixel experiment (revision-stage option). The unrelated StochNav draft in `paper/` is not touched. Planner checks: `experiments/cf_pilots/analysis/sign.py`, `gauge.py`.

## 1. Positioning

Literature check (2026-11-27, arXiv ids verified): 2611.24778 v1 (authors' repo still at 5c84bba; Thm 3.6 is minimax over separate-channel short-time worlds F_C and its lower bound uses a sign-unrecoverable rotation; authors mention that "a known physical mechanism may permit more accurate estimators"; open direction (iii) "rescale the residual with the command"). Follow-ups: 2611.28350 Fork-JEPA (paired fork outcomes add data), 2611.29259 Look Back (counterfactual returns as RL baselines), 2611.33745 WhatIf (replacing an individual effect by its conditional mean costs the conditional variance: the first term of our decomposition; cite). None relates residual reuse to BLUP / random-coefficient empirical Bayes, column-zero sign ambiguity, or saturation/mean ceilings.
Related work (verified): 2607.22430 (controlled-JEPA identifiability up to orthogonal transforms); 2609.05834; 2609.30264; 2510.08294 (dynamic-OT counterfactual identifiability); 2302.02228; 2301.09031; 2605.04413; 2603.27320 and 2507.12581 (Bodik cross-world rho for binary treatment: closest prior to the nested family; ours is the continuous-action identified-kernel generalisation); 2108.09816; 2109.03466; 2502.04654; 2505.01600; 1709.10193; 1905.06584; 2012.09092; Gaillac (Oxford JMP, posterior effects in linear random-coefficient models). To verify: 1811.06272, 1905.05824; classical: Hildreth-Houck 1968, Swamy 1970, Robinson 1991, Beran-Hall 1992, Beran-Feuerverger-Hall 1996, Hoderlein-Klemela-Mammen 2010, Anderson-Rubin 1956, Cramer 1936, Robbins 1956, Efron 2011, Attias 1999, Willsky-Jones 1976, Isermann 2006, Blanke et al. 2016. Record verification in `paper_cf/refs_verification.md`.

Contributions.
- C1 Reduction and nested family: with a known action basis psi(a), same-disturbance transport is prediction in a linear random-coefficient model; residual reuse ⊂ Hildreth-Houck BLUP ⊂ random-coefficient EB posterior mean, with closed-form excess risks and exact equality conditions (Thm 1).
- C2 Identification and gauge: without a basis, second moments fix G up to a continuous rotation field and for scalar outcomes the minimax rule is outcome-blind (Prop 3); with a basis, HH identifies symmetric cross blocks iff the products psi_j psi_k are linearly independent; antisymmetric parts (d >= 2) and the shared/fresh noise split stay unidentified (Prop 2); rotation counterexample where the HH lift is about 92 points below residual reuse.
- C3 Column-zero sign non-identifiability for gain actuators (Prop 4) and four resolutions.
- C4 Corrected nonlinear-headroom criterion (Thm 5) with exchangeability counterexamples and the saturation corollary.
- C5 Ceiling decomposition (Prop 6), generalising 2611.24778 eq. 16 to every linear L.
- C6 Benchmark evidence on the authors' simulators and protocol, with pre-registered predictions including zero-gain controls.

Non-claims: no new transport algorithm (PCK ties HH: results/cwk/STAGE2.md; IDT ties RC-EB GMM: results/k0/K0.md); HH, BLUP, EB, NPMLE, Tweedie are classical; d = 1 law identification is Beran-Hall / Hoderlein et al.; second-order gauge is Thm 2.2 at second order plus factor-rotation indeterminacy; command rescaling is the authors' direction (iii); no contradiction with Thm 3.6 (gain faults lie outside F_C); no contacts, Push-T, multi-step or persistence claim (echo route failed; appendix); known psi is privileged structural knowledge; information-limited cells are constructed.

## 2. Theory

Notation: Y(a) = mu(H,a) + G(H,a) xi + n; xi in R^r shared across actions, E[xi|H] = 0, Cov xi = I; n fresh with covariance N (shared action-free noise is an action-free column of G). Target Y*(a') = mu(a') + G(a') xi; residual r = Y(a) - mu(a); C(a) = G G^T, K(a',a) = G(a') G(a)^T. Basis assumption A-psi: G = sum_q psi_q(a) B_q(H), psi_0 = 1; b_q = B_q xi; Gamma_jk = E[b_j b_k^T | H].

Thm 1 (nested family). Rules T_RR = mu(a') + r; T_L = mu(a') + L r; T_BLUP with L* = K(a',a)(C(a)+N)^{-1}; T_EB = mu(a') + E[G(a') xi | r, H, a].
(i) R(L) = tr[C(a') - 2 L K(a,a') + L (C+N) L^T]; R(L) - R(L*) = tr[(L-L*)(C+N)(L-L*)^T]. (ii) R(EB) <= R(BLUP) <= min{R(RR), R(blind)}. (iii) RR = BLUP iff G(a')G(a)^T = C(a)+N; with N = 0 RR exact iff (G(a') - G(a)) xi = 0 a.s. (paper eq. 10, Prop 3.2), risk tr Var Delta. (iv) BLUP = EB iff E[G(a') xi | r] affine (Thm 5). (v) Rank one G = s(a)^T: signed location-scale rule; binary action + Gaussian copula gives Bodik's rho rule. Proof: expand square, L2 projection, substitute. Anchors: STAGE1 hand_affine100 hh_global 96.8, pck_rank1 97.4, rr_learned 2.7; hand_l0 hh_global 98.1 >= rr 97.6.

Prop 2 (HH identification and gauge). E[r r^T | H,a] = sum_jk psi_j psi_k Gamma_jk + N. (a) Gamma_00 + N, Gamma_jj, Gamma_jk + Gamma_kj identified iff {psi_j psi_k}_{j<=k} linearly independent. (b) Not identified: split of Gamma_00 + N (N = 0 safe default); for d >= 2 antisymmetric A_jk = (Gamma_jk - Gamma_kj)/2. (c) Identified set {Gamma_sym + A ⪰ 0}; K(a',a) = K_sym + sum_{j<k} (psi_j(a')psi_k(a) - psi_k(a')psi_j(a)) A_jk. (d) If Gamma ≻ 0 the set has interior, so K not identified wherever the antisymmetric factor is non-zero. (e) Sufficient: direction-separable channels (A = 0; holds in E1-E3); low rank pins A through PSD (proved for one scalar driver by 2x2 block minors). Synthetic evidence (gauge.py, n 200k, noise 0.05, R^2 %): channel syn2 oracle 80.1 / HH lift 80.1 / symmetric closed form 80.0 / RR 42.5; rotation (I ± aJ)xi 99.8 / -42.0 to -43.5 / 64 / 50; thrust misalignment theta J a 98.8 / 98.8 / -970 / -100; misalignment + offset 65.9 / 65.2 / 42.9 / -13. Basis collinearity: products of (1,x,x^2,x^3) and (1,x,|x|) are collinear (cond 1e17-1e20, sign.py), explaining F4/F6 drops as identification failures.

Prop 3 (second-order impossibility without a basis). For continuous R: A -> O(r), GR has the same C; Gaussian xi gives identical factual laws; K ranges over {G(a') Q G(a)^T : Q in SO(r)}. Corollary d = 1, r >= 2, Gaussian: K in [-|g(a')||g(a)|, +|g(a')||g(a)|]; minimax rule is outcome-blind mu(a'); RR and HH strictly worse in the worst case. Proof: path-connectedness of SO(r); least-favourable 50/50 prior on K and -K. Remark: factual laws fix only div(p v) of a transport velocity; OT/rank-preserving transport is one gauge choice (2210.06662, 2510.08294).

Prop 4 (column-zero sign non-identifiability). Y = mu + sum_k s_k(a) v_k xi_k + n, independent xi_k, xi_1 symmetric; Z = {s_1 = 0} separates A into A+ and A-; s~_1 = s_1 on A+, -s_1 on A-. (a) Same factual law for every a. (b) Across Z, K~ - K = -2 s_1(a') s_1(a) v_1 v_1^T; wrong-world BLUP excess risk tr[dL (C+N) dL^T], dL = -2 s_1(a')s_1(a) v_1 v_1^T (C+N)^{-1}. (c) Resolutions: basis with s_1 in span psi but |s_1| not; C^1 class with transversal zeros; E[xi_1^3] != 0; analyticity. Flat zeros (deadzone, e^{-1/a^2}) not resolved by C^1. Evidence: STAGE1 hand_affine100 flow 3.2, ls_unsigned -10.3, hh_global 96.8; dial_d2_l100 flow 15.7 vs HH 99.5; dial lambda 0.5 (zero at boundary) flow 95.9. sign.py (d = 1, uniform xi): truth a, basis (1,a) cross-zero 38.5 / same-side 90.1, basis (1,|a|) 15.6 / 90.1; truth |a|, basis (1,|a|) cross-zero 90.0 vs 69.1 for (1,a); equal factual second moments.

Thm 5 (nonlinear headroom, corrected). m(r) = E[G(a') xi | r]; R(BLUP) - R(EB) = E||m(r) - L* r||^2 >= 0, equality iff m affine. (a) Gaussian xi, n: zero headroom. (b) Saturation: N = 0, G(a) full column rank r <= d gives xi = G^+ r, all coincide (explains Stage 2 cells). (c) Row-space case G(a') = M G(a), Gaussian n with N ≻ 0: Tweedie m = M[r + N grad log p(r)], affine iff G(a) xi Gaussian (Cramer). Counterexamples: iid non-Gaussian xi, r = xi1 + xi2, G' = (1,-1) gives m = 0; G' = (1,0) gives r/2. Strict example: syn2 gated pi 0.15, exact Bayes 95.35 vs oracle BLUP 80.46 (K0.md). Genericity not claimed.

Prop 6 (ceiling decomposition). (a) Learned mean mu^ = mu + e (H-measurable): R(L; mu^) = R(L; mu) + E||e(a') - L e(a)||^2 (L = I is eq. 16). (b) With F_par ⊃ sigma(H, r) including the hidden persistent field: R(f) = irreducible + information + nonlinear headroom + estimation, for f affine in r. Empirical mapping: oracle_kernel* (exact mean, parent covariance), oracle_kernel_lm* (learned mean, parent map); F1: 98.3 -> 94.8 -> 94.0 means mean effect 3.5, kernel effect 0.8.

New vs known: known Thm 1 (i, ii, v algebra), Prop 2(a), Prop 3 gauge, continuity remark, Pythagoras/Tweedie/Cramer; new Thm 1(iii) and nesting account, Prop 2(b-e) and rotation counterexample, minimax-blind corollary, Prop 4, corrected Thm 5 with counterexamples and saturation corollary, Prop 6.

## 3. Experiments

Reused: replication (results/cwk/STAGE0.md, with the lambda = 1 literal miss disclosed); rank-one family (STAGE1.md); K0 synthetic (results/k0/K0.md); PCK = HH and echo failure (STAGE2.md, appendix); F7/F8 rows (results/design/cf4_idt.md, appendix).

Code changes:
1. cwk/sim.py: per-channel direction vector and axis key (gain on one actuator); kind `rotate` (out += scale eps J x on two axes). Old cells bit-identical (unit test).
2. cwk/worlds.py: per-driver law tanh (default) / gated(pi) = Bern(pi) U(-1,1) / gauss_clip; gates from a separate deterministic stream per (seed, parent, draw, driver), shared across counterfactual branches.
3. cwk/cells.py: INFOLIM cells E1o, E2o, E3o; UPSTREAM; MISALIGN; calibrated on simulator-only cohorts to outcome-blind oracle R^2 about 50 ± 8 (results/cwk/CALIBRATION.json new keys).
4. cwk/eb.py (new): rc_eb_gmm, glr_iso, quadrature (port from experiments/k0_run.py), batched Cholesky for d <= 3, R <= 5; rc_flow (RealNVP prior on HH coefficients, self-normalised IS with 4096 samples); rc_npmle (2000-atom grid, EM); cvae_mog optional.
5. cwk/pipeline.py arms: hh_binned_nug (N from {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} x tr C/d by held-out factual Gaussian likelihood); hh_nug_ref* (tuned on development counterfactuals); oracle_kernel_lm*, oracle_quad* (port diag4.py); hh_basis variants (1,x), (1,|x|), (1,x,x^3), (1,x,x^2,x^3) with cond(Phi^T Phi); oracle_struct_bayes* (per-parent finite-difference Jacobian w.r.t. drivers, exact Bayes by quadrature, report linearisation R^2); glr_iso; rc_eb_gmm (K 16, 32, best of 4 by held-out NLL); rc_flow; rc_npmle; idt row. Save per-parent losses.
6. Scripts experiments/paper_r1.py ... paper_r6.py, paper_report.py; outputs results/paper/R*.json/.md, logs results/paper/logs/.
7. results/paper/PROTOCOL.md committed before any run; sealed cell config drawn by seeded RNG after the protocol commit, hash recorded before data generation.
Statistics as Stage 2: 5 fit seeds; 512 parents x 64 draws x 3 queries; whole-parent x seed paired bootstrap, 2000 reps; Holm across environments.

Run list (about 45 job-hours):
- R0 unit tests (bit-identical old cells; shared gates; EB arms reproduce K0 within 0.3).
- R1 5-seed confirmation of F1-F4 on all 9 Stage 2 cells (cached data), full arm set incl. hh_binned_nug, hh_nug_ref*, rc_eb_gmm, basis variants on the sat cell, oracle_kernel_lm*, oracle_quad*.
- R2a data for information-limited cells per environment (E1o HandReach d = 2, E2o Ball-in-Cup d = 2, E3o FetchReach d = 3): P gated pi 0.2 r = d+1; T tanh r = d+1; G gauss_clip r = d+1; pi sweep {0.05, 0.1, 0.5, 1} at r = d+1; r in {d, d+2} at pi 0.2; P at iid50; 2 extra data seeds of P; 1 sealed cell.
- R2b baseline ladder on R2a cells (trained, rr_learned, ls_signed_nug, hh_binned, hh_binned_nug, glr_iso with factual-NLL lambda and lambda-oracle*, rc_eb_gmm, rc_npmle, rc_flow, cvae, idt, oracle*, rr_exact*, oracle_kernel*, oracle_kernel_lm*, oracle_struct_bayes*); 5 seeds for P, T, G, P-iid50, sealed; 3 seeds for sweep and extra-data cells.
- R2c validate oracle_struct_bayes* against full-simulation quadrature (E1o-P, 64 parents).
- R3 upstream cells (hand, cup, fetch, three iid50 variants, hand_load, hand_affine50, hand_affine100), no-harm arms.
- R4 sign ambiguity: synthetic d = 1 (s in {a, |a|, a|a|, e^{-1/a^2} sign}, symmetric vs skewed xi_1, 4 bases plus channel-free MLP-G head) and hand_affine100 / E1_nu50_gain_iid100 with psi (1,x) vs (1,|x|); factual NLL and cross-zero vs same-side R^2.
- R5 antisymmetric gauge: synthetic rotation / misalignment ± offset / channel worlds with HH lift R in {r, r+1, r+2}, 8 restarts; Ball-in-Cup misalignment cell (rotate channel plus offset).
- R6 predictor-accuracy sensitivity: iid100 cells with a predictor trained on 512 episodes (trained R^2 about 41%).
Order: R0 -> R1 || R2a -> R2b -> R3 -> R4, R5, R6 -> R2c.

Pre-registered predictions:
- P1 saturation: on Stage 2 cells, hh_binned_nug, rc_eb_gmm, oracle_quad* within 1.0 of oracle_kernel_lm*.
- P2 Gaussian control: |rc_eb_gmm - hh_binned_nug| <= 1.0 on G cells.
- P3 headroom: rc_eb_gmm - hh_binned_nug increases as pi decreases and as r - d increases; >= 0 at pi = 1 within CI.
- P4 sign: basis (1,|x|) matches (1,x) on held-out factual NLL within 0.01 nats and loses >= 10 points on cross-zero queries in hand_affine100.
- P5 no harm: on upstream additive cells every HH/EB arm within 1.5 of rr_learned.
- P6 rotation: HH lift below RR in the rotation world; within 1 of oracle in the misalignment world.
The paper is written whatever the outcome.

Fault-rate motivation: intermittent actuator faults (Isermann 2006; Blanke et al. 2016) as per-interval activation probabilities; full pi curves and dense controls reported.

Figures/tables: Fig 1 nested family and information sets (TikZ); Fig 2 ceiling decomposition bars (R1); Fig 3 R^2 vs pi per environment (R2b) with headroom vs r - d inset; Fig 4 gauge (R5a); Fig 5 sign (R4). Table 1 theory summary (new/known, anchor); Table 2 replication; Table 3 rank-one family; Table 4 Stage 2 5-seed (R1); Table 5 information-limited ladder (R2b); Table 6 K0; Table 7 gauge/misalignment (R5); Table 8 upstream no-harm (R3); appendix R6, R2c, PCK/echo negatives, sealed cell, collinearity.

## 4. Manuscript outline (paper_cf/main.tex, elsarticle review)
Front matter: highlights (5, <= 85 characters), keywords, CRediT (single author), data/code availability, competing interests.
1 Introduction (1.2k words); 2 Background and related work (1.3k); 3 Setup (0.8k); 4 Theory (3.0k, proofs in Appendix A); 5 Experimental protocol (1.0k, deviations disclosed); 6 Results (2.5k: replication; rank-one and sign; saturation and ceilings; information-limited; gauge; no harm and predictor sensitivity); 7 Discussion with a practitioner's decision list (additive -> RR; known gain basis -> HH with nugget; r > d intermittent -> RC-EB; no basis -> outcome-blind in worst case) and relation to Thm 3.6 (0.8k); 8 Limitations (0.4k); 9 Conclusion (0.2k). Appendices A proofs, B cells and calibration, C arms and hyperparameters, D negative results (PCK, echo, IDT, kappa attenuation), E additional tables, F reproducibility (ENV.md, upstream commit 5c84bba, MIT).

## 5. Referee risks
Classical components (claim the reduction and evidence; Table 1; cite Bodik and Gaillac prominently); tailor-made cells (protocol first, physics untouched, full sweeps, controls, authors' cells, sealed cell, extra data seeds); state level (Prop 6(a), R6, authors' own exact-mean failure); privileged psi (authors' "known mechanism"; misspecification and collinearity results; Prop 3); Stage 0 lambda = 1 miss (disclose with matched-predictor -6.8); tuned nugget (factual likelihood; privileged reference separate; Prop 2(b)); HH-lift gauge (Prop 2 + R5); Thm 3.6 (gain faults outside F_C); single data realisation (extra seeds); approximate oracles (R2c); persistence route (appendix negative); concurrent work (re-run arXiv search the day before submission); negativity (decision list, Prop 4 explains published failures).
