#!/bin/bash
# Budget sweep at harder goal offsets (TwoRoom, released LeWM checkpoint).
cd /content/le-wm
export STABLEWM_HOME=/content/swm_home MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for off in 50 100; do
 mkdir -p /content/out/sweep2/off$off
 bud=$((off*2))
 for N in 30 100 300 1000; do
  f=/content/out/sweep2/off$off/tworoom_N${N}_S30_seed0.json
  [ -f $f ] && continue
  python3 sel_eval.py --config-name=tworoom policy=quentinll/lewm-tworooms \
    eval.goal_offset_steps=$off eval.eval_budget=$bud world.max_episode_steps=$bud \
    eval.num_eval=50 seed=0 solver._target_=sel_lib.LoggedCEM solver.batch_size=25 solver.num_samples=$N \
    +solver.topk_frac=0.1 +out=$f > /content/out/sweep2/off$off/log_N${N}.txt 2>&1
 done
done
echo ALLDONE > /content/out/sweep2/DONE
