#!/bin/bash
cd /home/user/world_models/results/p4
for s in 0 1 2; do python3 pilot_a2.py $s > pilot_a2_s$s.log 2>&1; done
