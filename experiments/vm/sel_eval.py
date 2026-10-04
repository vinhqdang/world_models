"""Run LeWM evaluation with instrumentation. Usage (inside /content/le-wm):

    python sel_eval.py --config-name=tworoom policy=quentinll/lewm-tworooms \
        solver._target_=sel_lib.LoggedCEM solver.num_samples=300 solver.n_steps=30 \
        +out=/content/out/tworoom_N300_S30_seed0.json
"""
import importlib
import sys

import sel_lib

out = None
for a in list(sys.argv):
    if a.startswith('+out='):
        out = a.split('=', 1)[1]
        sys.argv.remove(a)
assert out, 'pass +out=<path>'
sel_lib.LOG.debug = any(a == '+debug=1' for a in sys.argv)
sys.argv = [a for a in sys.argv if a != '+debug=1']
import os
os.makedirs(os.path.dirname(out), exist_ok=True)

sel_lib.LOG.meta['argv'] = sys.argv[1:]
sel_lib.patch_criterion()
sel_lib.patch_policy()
sel_lib.patch_world(out)
import runpy
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval.py')
runpy.run_path(sys.argv[0], run_name='__main__')
