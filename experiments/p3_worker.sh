#!/bin/bash
# usage: p3_worker.sh gt|stats [rev]   -- lock-based worker so several can share the queue
export OMP_NUM_THREADS=1
cd /home/user/world_models
MODE=$1; REV=$2
list() { ls suite_hi/ckpt/*.pt results/p3/ckpt/*.pt; }
for pass in $(seq 1 200); do
  files=$(list); [ "$REV" = rev ] && files=$(list | tac)
  did=0
  for f in $files; do
    n=$(basename $f .pt)
    if [ "$MODE" = gt ]; then out=results/p3/gt/$n.json; else out=results/p3/stats/$n.json; fi
    [ -f $out ] && continue
    mkdir results/p3/locks/${MODE}_$n 2>/dev/null || continue
    did=1
    if [ "$MODE" = gt ]; then
      python experiments/d8_eval.py --ckpt $f --E 32 --N 16 --M 4 --iters 2 --H 10 --total_steps 400 --ep_len 120 --seed 901 --out $out > results/p3/logs/gt_$n.txt 2>&1
    else
      python experiments/p3_stats.py --ckpt $f --name $n > results/p3/logs/stats_$n.txt 2>&1
    fi
  done
  if [ $did = 0 ]; then [ -f results/p3/ckpt/QUEUE_DONE ] && break; sleep 30; fi
done
