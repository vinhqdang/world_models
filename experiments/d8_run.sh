#!/bin/bash
# usage: d8_run.sh JOBFILE ; each line: NAME MODELSEED EVALSEED STEPS EXTRA_ARGS...   (2 parallel workers)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
run() { name=$1; m=$2; es=$3; steps=$4; shift 4
  out=results/d8/${name}_m${m}_e${es}.json
  [ -f $out ] && exit 0
  python experiments/d8_eval.py --ckpt suite_hi/ckpt/es_s${m}.pt --seed $es --total_steps $steps --out $out "$@" > results/d8/${name}_m${m}_e${es}.log 2>&1; }
export -f run
cat $1 | xargs -P 2 -L 1 bash -c 'run "$@"' _
