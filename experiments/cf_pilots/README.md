# Exploratory pilots for counterfactual world models

Scratch pilots written during the design round for improving on arXiv 2611.24778 (see results/design/cf1_contact.md, cf2_beyond_additivity.md, cf3_chaining.md). Exploratory, mostly single seed, not pre-registered. Some scripts expect a local clone of the authors' code (github.com/galilai-group/counterfactual-WM, MIT) and data files that are not stored here.

- contact/: pre-contact transport on Push-T (state coordinates, authors' simulator) and on the wall world.
- additivity/: polarized cross-world kriging checks (analytic gain dial, HandReach-like, folds, pendulum, amortized head).
- chaining/: twin residual smoother on the boat field and on a persistent gain/load plant, hindsight and forward planning.
