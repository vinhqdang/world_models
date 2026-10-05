#!/bin/bash
# usage: d6_run_round.sh "<schemes>" "<seeds>"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
for seed in $2; do for sch in $1; do
  out=results/d6/cl_${sch}_s${seed}.json
  [ -f $out ] && continue
  python experiments/d6_eval.py --ckpt suite_hi/ckpt/es_s${seed}.pt --planner risk --stage_w 1.0 --failcost 1 --kappa 3.0 --variant cliff_hi \
    --E 16 --N 32 --M 8 --H 10 --iters 3 --total_steps 300 --ep_len 120 --lam 0.0 --shrink 0 --seed $seed --scheme $sch --out $out >> results/d6/cl_log.txt 2>&1
done; done
echo ROUNDDONE >> results/d6/cl_log.txt
