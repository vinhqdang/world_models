#!/bin/bash
cd /home/user/world_models/results/p4
while [ ! -f pilot_a_s0.json ]; do sleep 5; done
python3 pilot_a.py 1 250,1000 > pilot_a_s1.log 2>&1
python3 pilot_a.py 2 250,1000 > pilot_a_s2.log 2>&1
python3 pilot_b.py 0 random,dagger,disagree,crossregret > pilot_b_s0.log 2>&1
python3 pilot_b.py 1 random,dagger,disagree,crossregret > pilot_b_s1.log 2>&1
python3 pilot_b.py 2 random,dagger,disagree,crossregret > pilot_b_s2.log 2>&1
