#!/bin/bash
# Extra world-model variants with a spread of quality: training length and data coverage. Seeds 10+ (disjoint from suite_hi seeds 0-2).
export OMP_NUM_THREADS=1
cd /home/user/world_models
T() { # name kind steps seed n_data edge
  [ -f results/p3/ckpt/$1.pt ] && return
  python experiments/train_wm.py --kind $2 --variant cliff_hi --obs fixed --steps $3 --seed $4 --n_data $5 --edge $6 --out results/p3/ckpt/$1.pt > results/p3/logs/train_$1.txt 2>&1
}
for s in 10 11; do
  T det_k500_s$s det 500 $s 300000 0.25
  T det_k2000_s$s det 2000 $s 300000 0.25
  T gauss_k500_s$s gauss 500 $s 300000 0.25
  T gauss_k2000_s$s gauss 2000 $s 300000 0.25
done
for s in 10 11; do
  T es_k500_s$s es 500 $s 300000 0.25
  T es_k1500_s$s es 1500 $s 300000 0.25
  T es_k3000_s$s es 3000 $s 300000 0.25
done
for s in 10 11; do
  T det_low_s$s det 3000 $s 30000 0.0
  T gauss_low_s$s gauss 3000 $s 30000 0.0
  T es_low_s$s es 2000 $s 30000 0.0
done
echo done > results/p3/ckpt/QUEUE_DONE
