#!/bin/bash
# single sequential worker: reads results/n2/jobs.txt (lines: SCHEME MODELSEED EVALSEED STEPS); skips finished jobs; waits for new lines.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
idle=0
while true; do
  did=0
  while read -r scheme m es steps; do
    [ -z "$scheme" ] && continue
    out=results/n2/cl_${scheme}_m${m}_e${es}.json
    [ -f "$out" ] && continue
    extra=""; [ "$scheme" = "learned" ] && extra="--learned_path results/n2/learned_nodes_q.pt"
    python experiments/n2_eval.py --ckpt suite_hi/ckpt/es_s${m}.pt --scheme $scheme --seed $es --total_steps $steps --out $out $extra > results/n2/cl_${scheme}_m${m}_e${es}.log 2>&1
    did=1; break
  done < results/n2/jobs.txt
  if [ $did = 0 ]; then idle=$((idle+1)); [ $idle -gt 600 ] && exit 0; sleep 10; else idle=0; fi
  [ -f results/n2/STOP ] && exit 0
done
