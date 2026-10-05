# Prior-art review: feedback-parameterised sample-based planning with stochastic latent world models

Date: 2026-10-05. Searches: arXiv API (title/abstract queries; the API is flaky and returns nothing for long AND-queries), WebSearch, Crossref (classical references). "FT" = full text read by pdftotext (read selectively with grep plus passages, not end to end). "Abs" = abstract only. "Mem" = cited from memory, not retrieved in this session (unverified).

## Angle 1. Classical stochastic/robust MPC with feedback parameterisation

- Mayne, Seron, Rakovic, Automatica 2005, doi 10.1016/j.automatica.2004.08.019 (Crossref only; content from memory). Tube MPC: u = v + K(x - xbar) around a noise-free nominal. This is the same structural form as a_nom + K(z - zbar). It optimises the nominal; K is fixed, not optimised.
- Goulart, Kerrigan, Maciejowski, Automatica 2006, doi 10.1016/j.automatica.2005.08.023 (Crossref only). Optimising over affine state/disturbance feedback policies inside MPC. Nominal plus feedback optimised jointly (convex), with bounded disturbances.
- Skaf and Boyd, IEEE TAC 2010, doi 10.1109/tac.2010.2046053; Oldewurtel, Jones, Morari, CDC 2008, doi 10.1109/cdc.2008.4738806 (Crossref only). Affine disturbance feedback (ADF) in chance-constrained stochastic MPC.
- Lee and Borrelli, arXiv 2411.13935 (FT). ADF stochastic MPC whose online problem is "optimization over nominal inputs and a reduced set of learned feedback gains", scenario-sampled. This is the closest classical match to "low-dimensional K optimised jointly with a_nom under sampled noise". Linear systems only, convex programme, not a latent model.
- Bartos, Didier, Sieber, Zeilinger, arXiv 2502.06469 (FT skimmed). Stochastic MPC optimising disturbance-feedback matrices online, with guarantees. Linear.
- Scokaert and Mayne, IEEE TAC 1998, doi 10.1109/9.704989 (Crossref only). Min-max feedback MPC. Establishes that open-loop predictions are conservative and feedback predictions are better.
- Bertsekas, Dynamic Programming and Optimal Control (Mem). Open-loop feedback control (OLFC) vs closed-loop; the "value of feedback" is textbook. Bar-Shalom and Tse, IEEE TAC 1974, doi 10.1109/tac.1974.1100635 (Crossref only), name the same OLF distinction.
- Branch MPC, Chen, Rosolia et al., arXiv 2109.05128 (Abs). Plans over feedback policies as a trajectory tree with CVaR.
- Others (Mem or Crossref only): Mesbah 2016 (doi 10.1109/mcs.2016.2602087); Hewing, Wabersich, Zeilinger 2020 (doi 10.1016/j.automatica.2020.109095); Bernardini and Bemporad 2012 (scenario trees, doi 10.1109/tac.2011.2176429); Lucia et al. 2013 (multi-stage NMPC, doi 10.1016/j.jprocont.2013.08.008); Alsterda and Gerdes 2019 (contingency MPC, doi 10.23919/acc.2019.8815260); Okamoto, Goldshtein, Tsiotras 2018 (covariance steering, doi 10.1109/lcsys.2018.2826038).

Does not do: none uses a learned stochastic latent predictor, a decoder-free JEPA, CEM/sampling over (a_nom, K), or a planning-success benchmark. Does do: everything conceptual (nominal plus feedback, joint optimisation, scenario sampling, OL vs CL gap).

## Angle 2. Sampling-based planners with feedback

- Gupta and Moore, arXiv 2608.03978 (FT, grep plus passages). CEM trajectory optimisation under process noise (stochastic dynamics, also a learned NN model) with segments connected by local TVLQR feedback policies. Feedback gains are synthesised (TVLQR from rollout-fitted linearisation), not optimised by CEM; rollouts inside a segment are open-loop. Closest "CEM + stochastic dynamics + feedback" paper.
- Pan et al., Feedback Sampling MPC (FS-MPC), arXiv 2608.19443 (FT, grep plus passages). Argues open-loop rollouts diverge on unstable systems (sample cost grows with horizon) and samples through an optimised feedback policy (iLQR or RL gains). Deterministic dynamics; feedback is used to build proposals, gains come from iLQR/learned stabiliser, not optimised by the sampler. Same variance-growth motivation as our t sigma^2 claim.
- Belvedere et al., Feedback-MPPI, arXiv 2506.14855, RA-L 2026 (Abs plus web summary). Local linear gains from MPPI sensitivity applied between solves. Gains are post-hoc, not in the sampled objective.
- Gandhi et al., RMPPI arXiv 2102.09027; Yin et al., CC-MPPI arXiv 2109.12147 (FT skimmed); Balci et al., arXiv 2110.07744 (Abs). Tube/covariance-steering MPPI with an ancillary tracking controller. Feedback is outside the sampled objective.
- Williams et al., RSS 2018, doi 10.15607/rss.2018.xiv.042 (Crossref only). Tube-style robust sampling MPC (Mem for content).
- Wang and Ba, POPLIN, arXiv 1906.08649 (FT, passages). CEM over policy-network parameters (POPLIN-P) inside a PETS stochastic ensemble, with MPC execution (first action only). The planned object is a state-feedback policy evaluated at the model-sampled state. This is the structurally closest ML prior art: sample-based planning over a closed-loop parameterisation in a stochastic learned model. It differs in that the policy is a generic network, not nominal plus deviation gain, and the motivation is search smoothness, not OL/CL mismatch.

## Angle 3. Planning over policies with learned models

- Huh, Amortized Feedback Planning, arXiv 2609.35012, 2026-09-28 (FT, grep plus passages). States that optimising a fixed action sequence ignores observation response "even if the controller replans" and contrasts open-loop feedback (OLF) MPC (with CEM and L-BFGS solvers) against restricted feedback MPC (optimises belief-dependent tanh-affine actions) in a learned Gaussian belief model. The same OL-vs-CL claim as ours, just not for JEPA/latent predictors or energy-score models. Also 7 days old, so concurrent; the reviewer may know it.
- Deb, Wright, Banerjee, arXiv 2603.22430 (Abs). Inference-time optimisation of policy parameters by backprop through imagined rollouts of a differentiable world model.
- Chua et al., PETS, arXiv 1805.12114; Hafner et al., PlaNet, arXiv 1811.04551 (both in refs.bib; content from memory). Open-loop CEM over action sequences through sampled stochastic models, which is the exact baseline of our method.
- PILCO (Deisenroth and Rasmussen, ICML 2011), Dreamer actor (arXiv 1912.01603), TD-MPC/TD-MPC2 policy prior, Stochastic MuZero (ICLR 2022, not on arXiv) (all Mem/Abs). Policy search or tree search in stochastic models is closed-loop, but not within a planner over (a_nom, K).
- Lowrey et al., POLO, arXiv 1811.01848 (Abs only). Open-loop/closed-loop sample-based planning classics: Weinstein and Littman, AAAI 2013, doi 10.1609/aaai.v27i1.8547; Lecarpentier et al., IJCAI 2018, doi 10.24963/ijcai.2018/327 (Crossref only); Bubeck and Munos, COLT 2010 (Mem). Guided policy search (Levine and Koltun, ICML 2013; Levine and Abbeel 2014; Mem) uses time-varying linear-Gaussian controllers K_t(x - xbar_t) + k_t under learned models.

## Angle 4. Latent/JEPA planners with feedback or closed-loop rollouts

- Nath et al., SLS^2 / "Pixels to Proofs", arXiv 2606.15594 (FT, grep plus passages). In an action-conditioned JEPA latent model, optimises a nominal trajectory and closed-loop response matrices (system level synthesis; feedback on deviation from the nominal), with conformal bounded latent disturbances and tube-tightened constraints. Gradient/iLQR based, worst-case set, not stochastic particles, not CEM, not expected cost. It already is in refs.bib (nath2026sls); the current related.tex must be checked for how it is described. The closest paper in the latent-JEPA setting.
- Raghavan and Singh, arXiv 2609.02811 (Abs) and ARC-Bench, arXiv 2609.05461 (FT grep). Open-loop rollout evaluation vs closed-loop operation; ARC-Bench shows replanning masks ranking defects. Diagnostic, no method.
- Alrasheed et al., arXiv 2609.39235 (Abs). Pure imagination 23% vs MPC 30% success. Open-loop vs MPC, no feedback parameterisation.
- Chen et al., DREAM-Chunk, arXiv 2606.18589 (Abs). Stochastic latent model for reactive re-selection of action chunks under stochastic dynamics. Closest in spirit on "open-loop commitments are brittle under noise", but no feedback law.
- VJEPA arXiv 2601.14354 (FT grep): belief is a sufficient information state so the optimal policy depends on it, but it plans with sampling MPC; D-JEPA arXiv 2609.24749 (FT grep): no feedback or closed-loop terms found; LePlanner arXiv 2609.13845 (FT grep): amortised controller, deterministic.

## Angle 5. Risk-sensitive closed-loop planning

- Singh, Chow, Majumdar, Pavone, IEEE TAC 2019, doi 10.1109/tac.2018.2874704; Sopasakis et al., Automatica 2019, doi 10.1016/j.automatica.2018.11.022 (Crossref only; content Mem). Time-consistent risk-averse MPC with closed-loop policies (scenario trees, CVaR).
- Branch MPC (above): CVaR plus feedback policy as a trajectory tree.
- Yin et al., RAMPPI (already in refs.bib): CVaR with open-loop MPPI sequences.
- Rockafellar and Uryasev (CVaR), in refs.bib.
- Nothing found combining CVaR, feedback gains, and learned latent predictors. This is a gap, but a thin one.

## Verdict

Covered by prior art: (i) parameterising the plan as nominal plus feedback on deviation from a noise-free nominal (tube MPC, ADF, SLS, GPS, iLQG); (ii) optimising nominal and gains jointly under sampled disturbances (Lee and Borrelli for linear SMPC); (iii) open-loop prediction being worse than feedback prediction in expectation (min-max feedback MPC, OLFC, Bertsekas; and for learned models, Huh 2609.35012); (iv) CEM over a closed-loop policy parameterisation in a stochastic learned model with MPC execution (POPLIN-P); (v) feedback inside a latent JEPA planner (SLS^2, robust, gradient-based); (vi) feedback inside sampling-based MPC (Gupta and Moore, FS-MPC).

Not found: a sample-based (CEM) planner that jointly optimises a_nom and a low-dimensional K on the deviation between a sampled latent and the noise-free latent rollout of a stochastic, energy-score, decoder-free JEPA predictor, with replan-first-action execution and a closed-loop success/timeout/fall evaluation. Not found is not proven absent: arXiv search here is keyword-limited and the 2026 literature is large.

## What a hostile reviewer says

1. "Tube MPC and affine disturbance feedback, 1998 to 2020. The gain on deviation from a noise-free nominal is the tube controller. Novelty is zero at the formulation level."
2. "POPLIN-P and Gupta and Moore already do CEM over closed-loop parameterisations in stochastic learned models; Huh (Sept 2026) already makes your OL vs CL point."
3. "Open-loop planning is worse than closed-loop for every H>=2 is a textbook observation (OLFC). A one-dimensional absorbing-barrier proof is a toy."
4. "Feedback is virtual: only the first action executes, so K only changes how candidates are scored, not the executed policy. Gains of 0.827 to 0.908 on your own benchmark, plus more falls on the true simulator, say you traded timeouts for falls and may be exploiting the model."
5. "Compute and benchmark: 12% extra compute on one benchmark, no standard suite, no comparison against policy-parameter planning (POPLIN-P), a feedback-sampling baseline, or a closed-loop tree (Stochastic MuZero style)."

## Most defensible formulation

Do not claim feedback parameterisation, closed-loop prediction or the OL<CL inequality as new. Claim: "We transfer tube-style nominal-plus-feedback plan parameterisation, known in control, to sample-based planning with stochastic JEPA-style latent predictors. The feedback acts on the deviation of the sampled latent from the noise-free latent rollout, a quantity that exists only because the predictor is stochastic and decoder-free. We show (i) a controlled demonstration that scoring candidates open-loop while executing with replanning is mis-specified, with a simple-model analysis (absorbing boundary, LQG variance t sigma^2 vs bounded), and (ii) an empirical study at about 12% extra compute that separates timeout reduction from fall increase and reports the model-exploitation failure on the true simulator." Frame it as an empirical/transfer contribution with an explicit related-work paragraph; add baselines POPLIN-P-style policy-parameter CEM and fixed-gain (tube) feedback.

## Must-cite

Mayne et al. 2005; Goulart et al. 2006; Skaf and Boyd 2010; Oldewurtel et al. 2008; Scokaert and Mayne 1998; Bertsekas (OLFC); Bar-Shalom and Tse 1974; Lee and Borrelli 2411.13935; Bartos et al. 2502.06469; Todorov and Li 2005 (iLQG, doi 10.1109/acc.2005.1469949); Levine and Koltun 2013 (GPS); Gupta and Moore 2608.03978; Pan et al. 2608.19443; Belvedere et al. 2506.14855; Gandhi et al. 2102.09027; Williams et al. 2018; Wang and Ba 1906.08649; Huh 2609.35012; Nath et al. 2606.15594; Chen et al. 2109.05128; Singh et al. 2019; Sopasakis et al. 2019; Weinstein and Littman 2013; Lecarpentier et al. 2018; Antonoglou et al. (Stochastic MuZero, ICLR 2022); Chua et al. (PETS); Hafner et al. (PlaNet).

## Flags

Everything marked Crossref only or Mem has metadata verified only (Crossref) or is from memory. FT papers were read selectively. arXiv API queries with many AND clauses returned nothing, so absence is weak evidence. Huh 2609.35012 was found only via the arXiv API; venue unknown.
