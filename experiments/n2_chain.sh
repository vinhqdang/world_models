#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
python experiments/n2_walsh.py --model 0 --S 8 --N 16 --out results/n2/walsh_m0.json > results/n2/walsh_m0.log 2>&1
python experiments/n2_learn_nodes.py > results/n2/learn.log 2>&1
python experiments/n2_rank.py --schemes crn_anti,dir_oa,learned --noise_floor 0 --R 16 --S 24 --Mref 512 --out results/n2/rank_learned.json > results/n2/rank_learned.log 2>&1
python experiments/n2_walsh.py --model 1 --S 8 --N 16 --out results/n2/walsh_m1.json > results/n2/walsh_m1.log 2>&1
for p in $(pgrep -f "experiments/n2_eval.py"); do kill -CONT $p; done
touch results/n2/CHAIN_DONE
