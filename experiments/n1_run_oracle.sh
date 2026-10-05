#!/bin/bash
# oracle (true simulator) evaluation, E=100, 500 steps; seed-major so partial runs stay balanced; one process at a time
# usage: n1_run_oracle.sh "<seeds>" arm1 arm2 ...
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
seeds=$1; shift
for s in $seeds; do for arm in "$@"; do
  out=results/n1/oracle/${arm}_e${s}.json
  [ -f $out ] && continue
  python experiments/n1_eval.py --model oracle --arm $arm --E 100 --total_steps 500 --seed $s --out $out > /dev/null 2>&1
done; done
echo DONE >> results/n1/oracle_queue.done
