"""Smoke test of the paper pipeline on tiny sizes (not a result). usage: python3 experiments/paper_smoke.py [CELL]"""
import sys, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import upstream  # noqa
from cwk import cells, worlds, runner as RN, pipeline as PL
import dataclasses

name = sys.argv[1] if len(sys.argv) > 1 else "E1o_P"
w = cells.ALL_PAPER[name]()
t0 = time.time()
col = worlds.collect(w, 600)
pop = worlds.population(w, 12, 8, fd_h=0.5)
dev = worlds.population(dataclasses.replace(w, seed_offset=w.seed_offset + 5_000_000), 8, 8)
print("data", round(time.time() - t0), "s", flush=True)
cell = cells.pipeline_cell(w, col, pop, dev)
CFG = dict(epochs=8, width=64, batch=128, steps=100, n_seeds=5, threads=1, bins=2, cvae_steps=50, eb_steps=60, eb_Ks=(8,), eb_restarts=1,
           idt_steps=60, idt_restarts=1, flow_steps=60, flow_S=256, npmle_M=200, npmle_iters=10, npmle_n=2000, save_pk=True)
arms = ["hh_global", "hh_binned", "hh_binned_nug", "hh_nug_ref", "hh_basis_x", "hh_basis_abs", "hh_basis_x_x3", "hh_basis_cubic",
        "oracle_kernel_lm", "oracle_quad", "rc_eb_gmm", "rc_flow", "rc_npmle", "idt", "glr_iso", "glr_iso_lamo"]
res = PL.run_seed(cell, 0, arms, CFG)
det = RN.deterministic(cell, extras=True)
g = cell.primary
from cwk import metrics as M
var = M.target_variance(np.asarray(pop["truth"]), cell.groups[g])
for a, l in res["losses"].items():
    print(f"{a:18s} R2 {100 * (1 - l[g].mean() / var):7.2f}")
for a, l in det[g].items():
    print(f"{a:18s} R2 {100 * (1 - l.mean() / var):7.2f}  (deterministic)")
print({k: v for k, v in res["diag"].items() if not isinstance(v, dict)})
print("pk shape", np.asarray(res["pk_losses"]["rr_learned"][g]).shape, "total", round(time.time() - t0), "s")
