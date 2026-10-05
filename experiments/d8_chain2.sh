#!/bin/bash
cd /home/user/world_models
while pgrep -f "d8_chain.sh" > /dev/null; do sleep 10; done
experiments/d8_run_or.sh results/d8/jobs_or.txt
experiments/d8_run.sh results/d8/jobs_r4.txt
