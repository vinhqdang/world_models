#!/bin/bash
# learned-model evaluation, E=16, 480 steps per run, eval seed 400+model seed; model-major; one process at a time
# usage: n1_run_learned.sh "<arms>" "<model seeds>" [steps]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
steps=${3:-480}
for ms in $2; do for arm in $1; do
  out=results/n1/learned/${arm}_m${ms}_e$((400+ms)).json
  [ -f $out ] && continue
  python experiments/n1_eval.py --model learned --ms $ms --arm $arm --E 16 --total_steps $steps --seed $((400+ms)) --out $out > results/n1/learned/${arm}_m${ms}.log 2>&1
done; done
echo DONE >> results/n1/learned_queue.done
