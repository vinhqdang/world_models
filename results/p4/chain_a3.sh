#!/bin/bash
cd /home/user/world_models/results/p4
for s in 3 4 5; do python3 pilot_a.py $s 250 > pilot_a_s$s.log 2>&1; done
