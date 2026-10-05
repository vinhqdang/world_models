#!/bin/bash
cd /home/user/world_models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for ms in 0 1 2; do
  experiments/n1_run_learned.sh "ol_indep" "$ms" 520
  experiments/n1_run_learned.sh "ol_crn3 cf_crn3" "$ms" 480
done
for ms in 0 1 2; do
  python experiments/n1_diag.py --model learned --ms $ms --arms ol_indep,ol_crn3,cf_crn3,cf_indep,cf_crn3_b,cf_crn3_d,ref_crn64 --S 128 --reps 2 --out results/n1/diag_learned_m${ms}_s900.json > results/n1/diag_learned_m${ms}.log 2>&1
done
echo DONE >> results/n1/chain.done
