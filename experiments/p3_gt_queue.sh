#!/bin/bash
# Ground-truth closed-loop evaluation of every candidate model with ONE fixed planner (open-loop CEM, expected cost, N=16, M=4, iters=2, H=10),
# E=32 slots, 400 steps, eval seed 901 fixed before any offline statistic was computed.
export OMP_NUM_THREADS=1
cd /home/user/world_models
G() { # name ckptpath
  [ -f results/p3/gt/$1.json ] && return
  python experiments/d8_eval.py --ckpt $2 --E 32 --N 16 --M 4 --iters 2 --H 10 --total_steps 400 --ep_len 120 --seed 901 --out results/p3/gt/$1.json > results/p3/logs/gt_$1.txt 2>&1
}
for k in det gauss es; do for s in 0 1 2; do G ${k}_s$s suite_hi/ckpt/${k}_s$s.pt; done; done
for round in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  for f in results/p3/ckpt/*.pt; do n=$(basename $f .pt); G $n $f; done
  [ -f results/p3/ckpt/QUEUE_DONE ] && { for f in results/p3/ckpt/*.pt; do n=$(basename $f .pt); G $n $f; done; break; }
  sleep 60
done
echo done > results/p3/gt/GT_DONE
