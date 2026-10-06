#!/bin/bash
cd /home/user/world_models/results/p4
python3 pilot_b.py 2 random,dagger,disagree,crossregret > pilot_b_s2.log 2>&1
for s in 0 1 2; do python3 pilot_b.py $s random_local > pilot_bl_s$s.log 2>&1; done
