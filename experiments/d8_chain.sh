#!/bin/bash
cd /home/user/world_models
while pgrep -f "d8_run.sh results/d8/jobs_r1" > /dev/null; do sleep 10; done
experiments/d8_run.sh results/d8/jobs_r2.txt
experiments/d8_run.sh results/d8/jobs_r3.txt
