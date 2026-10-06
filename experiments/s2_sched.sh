#!/bin/bash
# usage: experiments/s2_sched.sh queue.txt
# Reads the queue file line by line (re-reading it, so lines can be appended while running), starts a line as soon as fewer than 3 evaluation
# processes run, stops at a line that says STOP. Lines whose output file already exists are skipped by the line itself.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/user/world_models
i=1
while true; do
  line=$(sed -n "${i}p" "$1")
  if [ "$line" = "STOP" ]; then break; fi
  if [ -z "$line" ]; then sleep 20; continue; fi
  while [ "$(pgrep -fc '^python experiments/s2_eval.py')" -ge 3 ]; do sleep 5; done
  nohup sh -c "$line" > /dev/null 2>&1 &
  sleep 3
  i=$((i+1))
done
wait
echo SCHED_DONE >> "$1.done"
