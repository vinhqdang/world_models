#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
run() { name=$1; seed=$2; shift 2; out=results/d8/oracle_${name}_e${seed}.json; [ -f $out ] && exit 0
  python experiments/d8_oracle.py --seed $seed --out $out "$@" > results/d8/oracle_${name}_e${seed}.log 2>&1; }
export -f run
cat $1 | xargs -P 2 -L 1 bash -c 'run "$@"' _
