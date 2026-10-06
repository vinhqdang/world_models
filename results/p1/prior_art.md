# P1 prior-art list (all ids fetched through the arXiv API and titles checked; about 70 keyword queries plus 9 web searches)

| id | title (short) | what it does |
|---|---|---|
| 2609.30264 | AD-WM: Action-Discriminative World Models | residual latent dynamics + inverse-dynamics and normalised action-recovery loss (CMI motivated) on predictor outputs, heads discarded at test |
| 2606.31232 | Delta-JEPA | decodes the action from the latent displacement; shows action-sensitivity analyses |
| 2608.17542 | Contrastive Inverse Dynamics for JEPA (AC-MTM) | Action-NCE on latent transitions as anti-collapse and action-sensitivity |
| 2610.03137 | Keeping JEPA World Models Plannable | one inverse-dynamics auxiliary loss on encoder and predicted latents restores action-sensitive probes |
| 2607.02403 | ACID | inverse-dynamics cycle consistency folded into the planning cost |
| 2607.26712 | ActSWM | "context collapse"; transition-separation constraint on rollouts |
| 2608.06706 | Dueling World Models | post-hoc advantage readout: subtract the mean over actions to isolate the action channel |
| 2607.18715 | DWM: Separating World Effects from Actions | auxiliary world head action-invariant + orthogonality constraint, additive decomposition |
| 2608.04653 | CoCo: Overcoming Statistical Bias in Action-Controllable World Models | counterfactual consistency (zero-action, inverse-action, mirrored) against statistical shortcuts |
| 2609.37378 | Do-JEPA | simulator-reset interventions, effect loss on z^a - z^ref |
| 2608.24885 | WorldEcho / WorldSync | diagnosis of off-expert action following; intervention-effect alignment |
| 2607.22430 | Identifiability of Controlled World Models | identifiability of controlled dynamics; counterfactual-error amplification scales inversely with the conditional action-excitation margin |
| 2609.31161 | I Act Therefore I Am | identifiability of causal states under JEPA action conditioning; needs sufficient action-induced variation |
| 2606.09028 | ATM | action-identifiability via Bayes inverse risk; real-vs-predicted inverse-transfer matrix tracks planning |
| 2609.05461 | ARC-Bench | frozen JEPA models rank candidate actions badly; replanning hides it |
| 2609.33030 | What Must a World Model Distinguish for Planning? | mechanism / response / decision sufficiency hierarchy |
| 2608.22197 | Capability Separation (world-action models) | irreducible action-specific error of predictors not conditioning on the candidate action |
| 2609.23753 | OnlineWM | active simulator queries + causality-aware objective |
| 2604.01985 | World Action Verifier | forward-inverse asymmetry for finding errors on suboptimal actions |
| 2609.22816 | FIRM-WM | factual + common-reset interventional branches |
| 1706.09523 | Hahn, Murray, Carvalho | regularisation-induced confounding; propensity covariate / orthogonal parametrisation (causal ML, not world models) |
| 1906.02120 | Dragonnet / targeted regularisation | neural nets for treatment effects (causal ML) |
| 2207.09705 | Residual action prediction (copycat) | policy-side residualisation of the action on history |
