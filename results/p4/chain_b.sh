#!/bin/bash
cd /home/user/world_models/results/p4
for s in 0 1 2; do python3 pilot_b.py $s random,dagger,disagree,crossregret > pilot_b_s$s.log 2>&1; done
