# world_models

Research code for a study of planning with learned latent world models.

Status: work in progress. Nothing here is a finished result yet.

## Layout

- `selwm/` - small PyTorch modules: toy environments, an MLP-ensemble dynamics model, a CEM planner that
  records every iterate (with KL from the prior), an online estimator of the optimism curve, and a synthetic
  planning world with a planted error field.
- `experiments/` - scripts. `toy_identification.py`, `synth_identification.py`, `synth_shift.py` run on CPU.
  `experiments/vm/` holds the instrumentation used with the public LeWorldModel checkpoints
  (`sel_lib.py`, `sel_eval.py`, sweep scripts); these run on a GPU machine with `stable-worldmodel` and
  the LeWM repository installed.
- `results/` - raw outputs of the runs above.

## Question under study

When a planner picks the lowest predicted cost among many candidate action sequences, how much of the
predicted improvement is realised in the environment, and how should the amount of search be controlled?

## Reproducing the CPU experiments

```bash
pip install numpy scipy matplotlib torch
python experiments/synth_identification.py
python experiments/synth_shift.py
```

## LeWM measurements

```bash
# inside a checkout of lucas-maes/le-wm with stable-worldmodel[env] dependencies, transformers<5
python sel_eval.py --config-name=tworoom policy=quentinll/lewm-tworooms \
    solver._target_=sel_lib.LoggedCEM solver.num_samples=300 +solver.topk_frac=0.1 +out=out.json
```
