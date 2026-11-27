"""Simulator job: python3 experiments/paper_gen.py CELL [CELL ...]  (one torch thread, as the cached Stage 2 data)."""
import sys
sys.path.insert(0, "/home/user/world_models/experiments")
import torch
torch.set_num_threads(1)
from paper_common import *
for n in sys.argv[1:]:
    ensure_data(n, log=lambda *a, **k: print(*a, flush=True))
    open(f"{LOGS}/ready_{n}", "w").write("ok")
    print("ready", n, flush=True)
