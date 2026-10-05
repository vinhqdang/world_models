#!/bin/bash
# runs the command lines of $1 in order, keeping at most 2 d8 python processes alive (also counts already-running ones)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
while read -r cmd; do
  while [ "$(pgrep -fc '^python experiments/d8_(eval|oracle)')" -ge 2 ]; do sleep 3; done
  bash -c "$cmd" &
  sleep 2
done < $1
wait
