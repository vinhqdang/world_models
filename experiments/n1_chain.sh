#!/bin/bash
cd /home/user/world_models
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while [ ! -f results/n1/oracle_queue.done ]; do sleep 10; done
for rho in 0.5 0.8 0.95 1.0; do
  python experiments/n1_diag.py --model oracle --arms cf_crn3 --S 1000 --reps 4 --rho $rho --out results/n1/diag_oracle_rho${rho}_s900.json > /dev/null 2>&1
done
experiments/n1_run_learned.sh "ol_indep ol_crn3 cf_crn3" "0 1 2" 480
for ms in 0 1 2; do
  python experiments/n1_diag.py --model learned --ms $ms --arms ol_indep,ol_crn3,cf_crn3,cf_indep,cf_crn3_b,cf_crn3_d,ref_crn64 --S 128 --reps 2 --out results/n1/diag_learned_m${ms}_s900.json > results/n1/diag_learned_m${ms}.log 2>&1
done
echo DONE >> results/n1/chain.done
