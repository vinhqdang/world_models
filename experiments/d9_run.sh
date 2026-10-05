#!/bin/bash
# usage: d9_run.sh "<arm list: name:fb:scheme ...>" "<model seeds>" [M] [tag-suffix]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
M=${3:-8}; SUF=${4:-}
for ms in $2; do for arm in $1; do
  IFS=: read name fb sch <<< "$arm"
  out=results/d9/${name}${SUF}_m${ms}.json
  [ -f $out ] && continue
  python experiments/d9_eval.py --ckpt suite_hi/ckpt/es_s${ms}.pt --fb $fb --scheme $sch --M $M --seed 211 \
    --E 16 --N 32 --H 10 --iters 3 --total_steps 450 --ep_len 120 --out $out > results/d9/${name}${SUF}_m${ms}.log 2>&1
done; done
echo DONE >> results/d9/queue_$$.done
