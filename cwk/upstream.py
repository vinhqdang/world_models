"""Makes the authors' repository (MIT licence, https://github.com/galilai-group/counterfactual-WM, commit 5c84bba)
importable as ``cwm``. It is used as a dependency and is never modified."""
import os
import sys
import warnings

UPSTREAM = os.environ.get("CWM_UPSTREAM", "/home/user/cwm_upstream")
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)
warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "disable")
