#!/bin/bash
cd /home/user/world_models/results/p4
while [ ! -f pilot_b_s2_random_dagger_disagree_crossregret.json ]; do sleep 20; done
for s in 0 1 2; do python3 pilot_b.py $s random_local > pilot_bl_s$s.log 2>&1; done
