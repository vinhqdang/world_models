#!/bin/bash
# Budget sweep on TwoRoom with the released LeWM checkpoint.
cd /content/le-wm
export STABLEWM_HOME=/content/swm_home MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p /content/out/sweep1
for seed in 0 1 2; do
 for N in 30 100 300 1000 3000; do
  f=/content/out/sweep1/tworoom_N${N}_S30_seed${seed}.json
  [ -f $f ] && continue
  python3 sel_eval.py --config-name=tworoom policy=quentinll/lewm-tworooms world.max_episode_steps=50 \
    eval.num_eval=50 seed=$seed solver._target_=sel_lib.LoggedCEM solver.batch_size=25 solver.num_samples=$N \
    +solver.topk_frac=0.1 +out=$f > /content/out/sweep1/log_N${N}_seed${seed}.txt 2>&1
 done
done
echo ALLDONE > /content/out/sweep1/DONE
