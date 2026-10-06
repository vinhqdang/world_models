#!/bin/bash
cd /home/user/world_models
export OMP_NUM_THREADS=1
if [ "$1" = "A" ]; then
python experiments/p1_pilot.py --seeds 0 1 2 3 4 --sigma 0.05 --N 3000 --methods mle cin affine idm idm_delta nce ridm orth oracle_int mix0.05 mix0.1 mix0.25 --lam 1 1 1 0.001 0.0001 0.0001 0.0001 --out results/p1/eval/main_A.json > results/p1/eval/main_A.log 2>&1
python experiments/p1_pilot.py --seeds 0 1 2 3 4 5 6 7 8 9 --sigma 0.02 --N 3000 --methods mle cin affine oracle_int mix0.1 --out results/p1/eval/sw_0.02.json > results/p1/eval/sw_0.02.log 2>&1
else
python experiments/p1_pilot.py --seeds 5 6 7 8 9 --sigma 0.05 --N 3000 --methods mle cin affine idm idm_delta nce ridm orth oracle_int mix0.05 mix0.1 mix0.25 --lam 1 1 1 0.001 0.0001 0.0001 0.0001 --out results/p1/eval/main_B.json > results/p1/eval/main_B.log 2>&1
python experiments/p1_pilot.py --seeds 0 1 2 3 4 5 6 7 8 9 --sigma 0.15 --N 3000 --methods mle cin affine oracle_int mix0.1 --out results/p1/eval/sw_0.15.json > results/p1/eval/sw_0.15.log 2>&1
python experiments/p1_pilot.py --seeds 0 1 2 3 4 5 6 7 8 9 --sigma 0.4 --N 3000 --methods mle cin affine oracle_int mix0.1 --out results/p1/eval/sw_0.4.json > results/p1/eval/sw_0.4.log 2>&1
fi
