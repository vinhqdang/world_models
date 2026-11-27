# Adversarial review of cf4_idt.md (Independent-Driver Transport)

Reviewer checks: `experiments/cf_pilots/idt/chk.py` (exact Bayes by per-activation-pattern quadrature; best-subset GLR fault isolation) and `experiments/cf_pilots/idt/gmm.py` (Hildreth-Houck lift plus a dependent full-covariance Gaussian-mixture prior; no independence, no ICA). Both reuse the syn2 generator.

## Verdict
NO-GO for the IDT kill test as a headline algorithm. At most a 3-CPU-hour pre-kill (K0). The independence/ICA step, which is the claimed novelty, buys at most about 1 R^2 point: a dependent non-Gaussian prior on the Hildreth-Houck coefficients recovers almost all of the F7 gain, and that prior is existing random-coefficient empirical Bayes.

## 1. Novelty (reviewer estimate 0.08-0.12 as an algorithm; about 0.15 for a corrected T2)
- Gaillac (2024), "Predicting Unobserved Individual-level Causal Effects" (Oxford job-market paper): linear random coefficients, one observation per unit, posterior effect E[Gamma | X, Y] by OT deconvolution plus Bayes and by a generalized Tweedie formula, used for policy counterfactuals. IDT's transport is x'^T PE(x, y); the only difference is the independence restriction.
- Jiang & Guntuboyina 2108.09816: NPMLE for random-coefficient models, arbitrary coefficient law, one observation per unit.
- Soloff, Guntuboyina & Sen 2109.03466: multivariate heteroscedastic NPMLE empirical Bayes.
- Attias 1999 (IFA) and Moulines et al. 0906.2885: the noisy MoG source model and posterior-mean reconstruction are IDT Stages 2-3 at fixed (H, a); Lewicki-Sejnowski 2000 for overcomplete inference.
- Liu, Moon & Schorfheide 1709.10193: Tweedie posterior means of unit effects under counterfactual scenarios.
- Beran-Hall 1992; Hoderlein-Klemela-Mammen 2010; Gaillac-Gautier 1905.06584, 2105.11720; Botosaru-Powell 2605.21367: identification of random-coefficient laws from one cross-section.
- Shi & Guo 1402.4239 (covariate-dependent ICA); Everson-Roberts 1999 (time-varying mixing).
- Fault isolation with signature directions: Willsky-Jones 1976 (GLR), Gertler 1998, Basseville-Nikiforov 1993, Kano et al. 2003, Lee-Yoo-Lee 2004.
- Counterfactual identifiability: 2106.08161, 2302.02228, 2301.09031, 2605.04413, 2605.13589, 1909.01525, 2401.14709.
- Under A4 (known psi, used by every working variant) IDT is Hildreth-Houck plus random-coefficient empirical Bayes with an IFA prior; independence is not needed for identification there. The ICA gauge collapse matters only without psi (Candidate 2), which failed in F6.
- Empirical check (gmm.py): HH lift plus dependent GMM prior, gated pi 0.15 seed 1: 92.98 (K 16), 93.35 (K 32) vs IDT 94.2-94.7 vs exact Bayes 94.13; tanh seed 0: 89.28 vs IDT 89.5 vs exact Bayes 89.3.
- Candidate 3 is Hoderlein-Mammen 2007 / Gaillac's generalized Tweedie formula (novelty below 0.1).

## 2. Correctness
- T1: correct, not new (2611.24778 Thm 2.2 at second order plus factor-rotation indeterminacy); a remark.
- T2: wrong as stated. (i) Gain actuators give columns a_k v vanishing on the hyperplane a_k = 0, which disconnects the action box; with symmetric sources a_k v and |a_k| v give identical factual laws but cross-world correlation +1 vs -1; continuity cannot fix the sign (needs C^1 transversal zeros or analyticity). This is the main case, including the pilot generator (syn2.py:12-14) and the proposed cells. (ii) Inseparability holds wherever columns are collinear (codimension d-1, i.e. curves for d = 2), not only g_i = +-g_j. (iii) With finitely many actions, signed permutations at different actions are not linked; with iid symmetric drivers the coupling is unidentified up to the hyperoctahedral group. (iv) H-binning gives a mixture of ICA models; heterogeneity across H produces non-Gaussianity even with Gaussian drivers, so the Stage 4 safeguard can fire spuriously. (v) Noise semantics inconsistent (shared n requires adding E[n | r]).
- T3: false as stated. "Linear iff Gaussian" holds for the factual signal or when G(a') = M G(a); for r > d counterexamples exist: G(a) = (1, 1), G(a') = (1, -1) gives E = 0 by exchangeability; G(a') = (1, 0) gives E[xi1 | xi1 + xi2] = r/2. Correct version: positive excess risk iff E[G(a')xi | r] is not affine in r; genericity needs proof. Bayes-rule dominance is trivial.
- T4: correct, trivial.

## 3. Pilot audit
- No leakage; oracle linear BLUP correctly specified.
- "Learned IDT beats particle Bayes" is a broken reference: syn2.py:31 uses 4000 prior particles with a likelihood width floored at 0.02; about 61% of particles sit at the origin for the gated prior. Exact quadrature Bayes (pi 0.15, noise 0.05, seed 1) is 94.13 vs linear BLUP 79.28; learned IDT matches it.
- Generator-model match is real (affine G with exact psi; MoG represents spike-and-slab and bimodal tanh almost exactly); tanh result flips 89.5 -> 80.1 between location and scale mixtures.
- Cheap baseline: best-subset GLR isolation with true signatures 88.4-91.5 (lambda 2-12), closing about 82% of the gap; with rotation-arbitrary HH columns about 69.
- F1-F4: no leakage; oracle_kernel_lm* is the honest ceiling; hh_binned_nug must be the reference baseline; iid50 cells must still be reported.

## 4. Benchmark legitimacy
As specified the cells are tailor-made. To defend: fault rates and amplitudes from the fault-diagnosis literature (Isermann 2006; Blanke et al.); full curves over pi in {0.05, 0.1, 0.2, 0.5, 1} and r in {d, d+1, d+2} with dense and Gaussian controls; a sealed, hash-committed third-party cell; upstream authors' cells as no-harm; temporally persistent faults.

## 5. Gate
P3 is uninformative. Mandatory arms: rc_eb_gmm (HH lift plus K in {16, 32} full-covariance GMM prior), rc_npmle, rc_flow (RealNVP prior, HH decoder), glr_iso, sparse_l1, cvae_mog, oracle_bayes* by quadrature, idt_nopsi.
- K0 pre-kill (about 3 CPU-hours): syn2 generator, gated pi in {0.15, 0.3} and tanh, 5 seeds; idt vs rc_eb_gmm (K 32, 3000 steps, lr 1e-2 cosine, batch 1024, best of 4 restarts by held-out factual NLL). NO-GO if mean idt - rc_eb_gmm < +1.5 or CI includes 0 in 2 of 3 settings.
- K1 (only if K0 passes): E1o/E2o/E3o x {gated 0.2, tanh} at iid100 (iid50 secondary), pi sweep, Gaussian control, Stage 2 and upstream no-harm cells, a sealed cell; 5 fit seeds; P1 idt - best non-privileged >= +2.0 (CI > 0) in >= 2/3; P1b idt - rc_eb_gmm >= +1.5 (CI > 0) in >= 2/3; P2 tanh +1.0; P4 no-harm deficit <= 1.0 (upper CI < 1.5); P5 forecast; P6 idt_nopsi within 3 points; sanity learned <= oracle_bayes*.

## 6. Fallback
Analysis paper: "When Can Counterfactual World Models Beat Residual Reuse? Saturation, Gauge, and Empirical-Bayes Limits of Same-Disturbance Transport". Contributions: saturation result and corrected ceilings; reduction of same-disturbance transport under a known basis to Hildreth-Houck plus random-coefficient empirical Bayes (nested family residual reuse / kriging / posterior mean with optimality conditions); corrected theory (T1, correct T3 with counterexamples, column-zero sign non-identifiability for gain actuators, Helmholtz statement); benchmark protocol with the full baseline ladder. No new transport algorithm claimed.
