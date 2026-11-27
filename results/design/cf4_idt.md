# Design round 3: beyond Hildreth-Houck (Independent-Driver Transport)

Exploratory pilot scripts and logs: `experiments/cf_pilots/idt/` (fit seed 0 unless stated; cached Stage 2 data; CPU). Not pre-registered.

## Bottom line of the designer
- No candidate clearly reaches novelty probability 0.3; the best (IDT) is estimated at 0.25-0.30 (an adversarial reviewer may say 0.15-0.20).
- The Stage 2 cells cannot carry any new transport algorithm: they are saturated to within about 1 R^2 point once the ceiling is computed with the learned mean.
- In a regime the Stage 2 cells do not cover (more independent drivers than scored coordinates, non-Gaussian drivers), IDT beat the exact-moment linear oracle by 9.5-15 points in a synthetic pilot, learned from single-action data. Not yet tested on the authors' simulators.

## Pilot findings
F1. The oracle-kernel headroom is a mean effect, not a kernel effect.

| cell | oracle_kernel* (exact mean) | same map, learned mean | hh_binned | per-parent quadratic oracle, exact mean |
|---|---|---|---|---|
| E1_nu50_gain_iid50 | 98.3 | 94.8 | 94.0 | 98.5 |
| E1_nu50_sat_iid50 | 95.2 | 85.5 | 79.0 | 95.5 |
| E1_nu50_gain_iid100 | 97.2 | 94.9 | 93.7 | 97.5 |

F2. RMS mean error / within-parent sd is 0.29 (gain) and 0.33 (sat) at iid50 but 0.15 at iid100; a 5-member ensemble has spread 0.07 and the same error. The oracle's exact mean conditions on the parent's hidden persistent field, which no predictor of H can know: oracle_kernel* is unattainable at iid50.

F3. No nonlinear headroom in these cells (quadratic oracle +0.2): with r = d = 2 abduction is exact linear inversion.

F4. The misspecified-channel gap was mostly a ridge artifact: on the sat cell, linear-psi Hildreth-Houck with a nugget of about 0.03 gives 84.3-84.7; basis (1, x, x^3) at tau 1e-3 gives 85.6 (ceiling 85.5); basis (1, x, x^2, x^3) drops to 65.3.

F5. Ensemble mean and epistemic kriging: at most +0.4. Dead.

F6. Channel-free non-Gaussian flow (unknown psi, noise-free square likelihood) failed (38-74 vs 94). The noisy mixture-of-Gaussians version with Hildreth-Houck initialisation only matches the ceiling:

| basis | gain K=1 | gain K=4 | sat K=1 | sat K=4 |
|---|---|---|---|---|
| linear | 94.4 | 94.2 | 84.4 | 85.6 |
| (1, x, x^3) | 94.3 | 94.7 | 29.6 | 29-30 |
| (1, x, x^2, x^3) | 85.2 | 83.3 | 80.0 | 83.3 |

Held-out factual NLL drops with K = 4 (2.18 to 2.12; 2.38 to 2.33): drivers are measurably non-Gaussian (kurtosis -1.38), but this buys at most 3 points.

F7. Synthetic overcomplete pilot (d = 2; common offset plus one gain per actuator; 80k factual samples; noise sd 0.05; single-action data; Hildreth-Houck initialisation; no mean error):

| drivers | r | oracle linear BLUP | HH + BLUP | learned IDT | particle Bayes, true model |
|---|---|---|---|---|---|
| gated 15% (3 seeds) | 3 | 79.5-80.9 | 79.3-80.8 | 94.2-94.7 | 91.5-91.8 |
| gated 30% (2 seeds) | 3 | 79.4 | 79.3-79.7 | 89.5-89.9 | 87.7-87.9 |
| gated 50% | 3 | 79.8 | 79.8 | 85.2 | 84.9 |
| gated 15%, noise 0.1 | 3 | 78.4 | 78.2 | 92.4 | 90.7 |
| tanh(2N), location mixture K=4 | 3 | 80.1 | 80.0 | 89.5 | 89.3 |
| tanh, zero-mean scale mixture | 3 | 80.1 | 80.0 | 80.1 | 89.3 |
| dense uniform | 3 | ~80 | 80.3 | 80.3 | 81.0 |
| gated 15% | 2 | 95.2 | 94.6 | 98.2 | 97.8 |
| tanh | 2 | 95.6 | 95.6 | 95.8 | 97.3 |

(The particle reference uses 4000 particles and is crude, hence below the learned model in some rows.)

F8. Tangent-quantile identity (scalar Y = xi0 + xi1 x, x in [0.2, 1]): Gaussian 95.7 vs oracle BLUP 95.7; tanh 97.2 vs 95.7; sparse 97.0 vs 95.7; dependent non-Gaussian 99.0 vs 98.9. Rank-preserving transport 95.4 / 96.8 / 96.1 / 98.9.

Consequences: add the reference arm oracle_kernel_lm* (linear oracle map with the learned mean), add nugget-tuned Hildreth-Houck (hh_binned_nug) as a baseline, make iid100 cells primary.

## Candidate 1 (recommended, conditional): Independent-Driver Transport (IDT)
Idea. The disturbance is a set of independent non-Gaussian physical drivers, each with its own action-dependent signature g_j(H, a). Independence plus non-Gaussianity collapses the gauge of 2611.24778 Thm 2.2 from a rotation field to signed permutations, which continuity in the action removes. The counterfactual is the closed-form nonlinear posterior mean: infer which driver(s) fired from the factual residual and move each along its own signature. Every linear rule, including kriging with exact moments, is strictly suboptimal when abduction is information-limited (r > d, noise, near-singular loadings).

Model. Y(a) = mu(H, a) + G(H, a) xi + n, G in R^{d x r}, r may exceed d. A1 xi_j independent given H, independent of a, mean 0, variance 1, non-Gaussian. A2 n ~ N(0, N), N diagonal. A3 G continuous in (H, a); columns pairwise non-collinear off a closed set with connected complement. A4 (optional prior) G = sum_q psi_q(a) G_q(H), psi = (1, a/a_lim). Counterfactual: same xi and n (n = 0 at transport). Data: one randomised action per history.

Algorithm.
- Stage 0: predictor.crossfit; standardised residuals (kernels.prepare).
- Stage 1: kernels.hh_moments then kernels.bm_lift (rank R, per 8 k-means context bins or global, 4 restarts).
- Stage 2: independent-factor likelihood. G_q per bin (or a head initialised to the lift); each source a K-component Gaussian mixture, re-standardised each step; diagonal noise floor 1e-3. Loss: exact marginal NLL, r ~ sum_{c in [K]^R} w_c N(G m_c, G S_c G^T + N), batched Cholesky. Defaults K = 4, means initialised at linspace(-1.2, 1.2); R in {d, d+1, d+2} by held-out factual NLL; Adam 1e-3 cosine, 3000 steps, batch 512, clip 5; early stopping on 10% held-out episodes; best of 4 restarts; K^R <= 256 (K = 3 for R >= 5).
- Stage 3: transport. rho_c proportional to w_c N(r; G m_c, Sigma_c); xi_hat = sum_c rho_c [m_c + S_c G^T Sigma_c^{-1}(r - G m_c)]; y_hat(a') = mu(a') + sigma G(H, a') xi_hat. The mixture also gives the counterfactual predictive law.
- Stage 4: factual safeguards. Non-Gaussianity gain = NLL(K=1) - NLL(K) on held-out data; below 0.01 nats fall back to K = 1 (nugget-aware Hildreth-Houck BLUP). Model-implied gain over BLUP reported before counterfactuals are seen.
- Code: kernels.fit_head kind "idt"; kernels.bayes_transport; pipeline arms idt, idt_k1, hh_binned_nug, sparse_l1, hh_ica2, cvae_mog; references oracle_kernel_lm*, oracle_bayes*; cwk/sim.py channel keys "mask" and "gate".

Known parts versus claimed new. Known: independent factor analysis (Attias 1999), overcomplete ICA, Hildreth-Houck moments, Beran-Hall 1992 (for d = 1, known psi, IDT is Beran-Hall plus empirical-Bayes posterior mean). Claimed new: the cross-world identification theorem (r > d, unknown channel shape), the nonlinear closed-form transport with strict dominance over every linear rule, the factual-only gain forecast, and beating the exact-moment linear oracle.

Theorems.
- T1 (second-order impossibility, proved). With G only continuous in a, the factual covariance function fixes G up to a continuous rotation field R(a); K(a, a') off the diagonal is unconstrained beyond positive-definiteness. Hildreth-Houck identification comes from the global parametric basis.
- T2 (gauge collapse; complete for r <= d, partial for r > d). Per (H, a) the residual law is a noisy linear ICA model; columns identified up to sign and permutation (Comon; Kagan-Linnik-Rao; for r > d Eriksson-Koivunen 2004, Wang-Seigal 2401.14709). Labelings can swap only where g_i = +-g_j (codimension d); continuity on the connected complement fixes the signed permutation, which leaves the joint law of (Y(a), Y(a')) unchanged. The cross-world coupling is identified without psi. Conjectured: source marginals for r > d; sign tracking through isolated column zeros.
- T3 (strict dominance, proved for N > 0). Tweedie: E[G xi | r] = r + N grad log p(r), linear iff p Gaussian iff (Cramer) every active driver is Gaussian; so the best linear rule has strictly positive excess risk generically. For N = 0, r <= d, full rank, all rules coincide (explains the saturated Stage 2 cells).
- T4 (reductions, proved). Gaussian sources: nugget-aware Hildreth-Houck BLUP; rank one: signed location-scale; action-free G: residual reuse.
- Not proved: finite-sample rates; dependent drivers.

Prior art (designer-verified ids): 2611.24778; Attias 1999 (Neural Comput. 11:803); Beran-Hall 1992 (Ann. Stat. 20:1970); Hoderlein-Klemela-Mammen 2010 (Econ. Theory 26:804); 2401.14709; 2206.13668; 2501.06777; 2302.02228; 2301.09031; 1805.08651; 1907.04809; 2106.08161; 1811.06272; 2006.06485; 2603.27320; 2507.12581; 1102.5721; 2601.07059; 0907.2451; 1902.05261; 2105.11720; 1909.08434; 2302.14380; Lewbel-Pendakur 2017 (JPE 125:1100); 2605.04413; 2108.13025; 2301.07755; 2108.05858; 2611.33745; 2611.29259; 2611.28350; 2609.05834. Not re-verified: Hildreth-Houck 1968, Swamy 1970, Beran-Feuerverger-Hall 1996, Masten 2018, Matzkin 2003, Chernozhukov-Hansen 2005, Athey-Imbens 2006, Heckman-Smith-Clements 1997, Comon 1994, Eriksson-Koivunen 2004, Kagan-Linnik-Rao 1973, Lewicki-Sejnowski 2000.
Novelty 0.25-0.30. Risks: "IFA plus Beran-Hall"; gains only in information-limited regimes, so new cells may be called tailor-made; driver independence untestable beyond factual fit; simulator gains may shrink under mean error.

Kill test (designer's proposal).
- New cells (authors' simulators, physics untouched), iid100 primary and iid50: E1o HandReach, common offset (+1, -1) plus one gain per wrist actuator (weights 0.4/0.3/0.3), r = 3 > d = 2; E2o Ball-in-Cup, offset plus per-actuator gains, r = 3, d = 2; E3o FetchReach, per-axis goal gains plus gripper load, r = 4 > d = 3. Two driver laws each: tanh and gated (activation 0.2). Amplitudes calibrated on simulator-only cohorts to oracle R^2 about 50%. Control: E1o with clipped Gaussian drivers. No-harm: the six Stage 2 cells.
- Arms: trained; rr_learned; hh_binned; hh_binned_nug; idt_k1; sparse_l1; hh_ica2; pck_pol; pck_joint; cvae; cvae_mog; idt; references oracle*, rr_exact*, oracle_kernel*, oracle_kernel_lm*, oracle_bayes*.
- Pass (5 fit seeds, 512 x 64 x 3, Holm): P1 gate cells iid100, idt - best non-privileged baseline >= +4.0, CI lower bound > +2.0, in >= 2/3 environments (stop if fails). P2 tanh cells iid100, >= +3.0, CI > 0, >= 2/3 (else restrict to intermittent faults). P3 idt >= oracle_kernel_lm* + 2.0 on gate cells in >= 2/3. P4 no harm: idt >= hh_binned_nug - 1.0 on Stage 2 cells and Gaussian control, deficit CI upper bound < 1.5. P5 factual gain forecast within max(2 points, 40%) on new cells.
- Expected: about half the synthetic gains. Compute about 2 CPU-days.

Paper path. Working title "Which Fault Fired? Identifying and Transporting Independent Disturbances in Counterfactual World Models". Contributions: saturation analysis with corrected ceilings; T1-T3; IDT with safeguards; overcomplete and intermittent-fault cells; hindsight planning; pixels with frozen encoders on a T4 only after state-level pass. Non-claims: not the first IFA; d = 1 with known psi is Beran-Hall plus empirical Bayes; no gain in square well-conditioned noise-free cells; no contact solution.

## Candidate 2: channel-free independent-driver head (ablation, not headline)
Drop psi, G(H, a) an MLP identified by T2 alone; noisy mixture likelihood, spline-Hildreth-Houck distillation init, smoothness penalty 1e-3 ||dG/da||^2. Evidence negative to neutral (F6). Novelty 0.25 for the theorem, about 0.15 as an algorithm. Kill test: sat, deadzone, sign-asymmetric gain cells at iid100; pass if >= best Hildreth-Houck basis variant + 3 in 2/3. Expected to fail.

## Candidate 3: tangent-quantile (action-flow) transport (scalar; theory)
Fit F(y | H, a) and move along the conditional-quantile tangent: y_hat(a') = y + sum_k (psi_k(a') - psi_k(a)) v_k, v = -(dF/dpsi_k)/(dF/dy). Exact Bayes rule for scalar Y when the response is linear in psi for fixed disturbance (a corollary of Hoderlein-Mammen 2007). Helmholtz statement for d >= 2: factual laws determine the action-momentum field p(y) E[dY/da | Y = y] up to its divergence-free part; the optimal-transport / Action Matching (2210.06662) gauge is wrong for rotational couplings. Evidence F8 (+1.3-1.5 over oracle BLUP). Novelty 0.20. Use as theory section and scalar baseline.

## Recommendation
Confirm F1-F4 over 5 seeds (about 2 CPU-hours), then run Candidate 1's kill test with Candidate 2 as ablation and Candidate 3 as scalar baseline. If P1 fails, fall back to the analysis paper (strengthened by the saturation result, corrected ceilings, T1, T3, Helmholtz statement). Open decision: whether new overcomplete and gated cells are an acceptable basis for the headline claim.
