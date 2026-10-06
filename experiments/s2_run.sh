#!/bin/bash
# usage: experiments/s2_run.sh jobs.txt   (runs the lines of jobs.txt, at most 3 at a time; finished outputs are skipped)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
xargs -P 3 -d '\n' -I{} sh -c "{}" < "$1"
echo ALLDONE >> "$1.done"
