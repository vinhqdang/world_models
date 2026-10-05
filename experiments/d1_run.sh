#!/bin/bash
# usage: d1_run.sh NAME CKPTKIND SEED [extra eval_stream flags...]; writes results/d1/NAME_s{SEED}.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
name=$1; kind=$2; seed=$3; shift 3
planner=risk; [ "$kind" = det ] && planner=mean
cd /home/user/world_models
python experiments/eval_stream.py --ckpt suite_hi/ckpt/${kind}_s${seed}.pt --variant cliff_hi --planner $planner --lam 0 --E 16 --N 32 --M 8 --H 10 --iters 3 \
  --total_steps 520 --shift_at 200 --shift_wind 0.13 --seed $seed --device cpu --out results/d1/${name}_s${seed}.json "$@" > results/d1/${name}_s${seed}.log 2>&1
