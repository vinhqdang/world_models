#!/bin/bash
# final-config chain (config A): det/mse x3 seeds, gauss/nll seed 0, es seed 1
F="--tta_gate 3 --tta_window 25 --tta_replay 0.5 --tta_params out --tta_lr 1e-3 --tta_steps 2 --tta_buflen 100"
cd /home/user/world_models
for s in 0 1 2; do experiments/d1_run.sh A_det det $s --tta mse $F; done
experiments/d1_run.sh A_gauss gauss 0 --tta nll $F
experiments/d1_run.sh A_es es 1 --tta es $F
