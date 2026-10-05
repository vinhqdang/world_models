#!/bin/bash
# Baselines under the same protocol as results/d9 (eval seed 211, 450 steps, E=16, N=32, M=8, H=10, iters=3)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
C="--variant cliff_hi --E 16 --N 32 --H 10 --iters 3 --total_steps 450 --ep_len 120 --seed 211 --device cpu"
for ms in $1; do
  [ -f results/main/det_mean_m$ms.json ] || python experiments/eval_stream.py --ckpt suite_hi/ckpt/det_s$ms.pt --planner mean --M 1 $C --out results/main/det_mean_m$ms.json > /dev/null 2>&1
  [ -f results/main/gauss_exp_m$ms.json ] || python experiments/eval_stream.py --ckpt suite_hi/ckpt/gauss_s$ms.pt --planner risk --lam 0 --M 8 $C --out results/main/gauss_exp_m$ms.json > /dev/null 2>&1
  e=suite_hi/ckpt/det_s$ms.pt,suite_hi/ckpt/det_s$(( (ms+1)%3 )).pt,suite_hi/ckpt/det_s$(( (ms+2)%3 )).pt
  [ -f results/main/ens3_m$ms.json ] || python experiments/eval_stream.py --ckpt $e --planner ensemble --M 9 $C --out results/main/ens3_m$ms.json > /dev/null 2>&1
done
echo DONE > results/main/queue_$2.done
