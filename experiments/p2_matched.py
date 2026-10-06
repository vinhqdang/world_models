"""Compute-matched baselines (10000 optimiser steps = budget of the self-calibrated arm: 4000 + 3x2000).
usage: p2_matched.py <system> <seeds>"""
import sys, json, torch
sys.path.insert(0, '/home/user/world_models'); torch.set_num_threads(1)
from selwm.p2_dyn import *; from selwm.p2_train import *
name = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]
cfg = json.load(open(f"/home/user/world_models/results/p2/{name}_selected.json"))
out = f"/home/user/world_models/results/p2/{name}_matched.jsonl"
for seed in seeds:
    d = Data(name, seed=seed, n_train=100, role="test")
    T = local_tangent_normal(d.train.reshape(-1, D_OBS), d.S["dim"], k=32)
    for tag, kw in [("iso_10k", dict(mode="iso", sigma=cfg["iso"])), ("normal_10k", dict(mode="normal", sigma=cfg["normal"], tangents=T)),
                    ("ss_10k", dict(mode="ss", ss_p=cfg["ss_p"], H=8))]:
        m = fit(d, seed=seed, steps=10000, bs=1024, lr=2e-3, **kw)
        r = evaluate(m, d)
        with torch.no_grad():
            te = d.test; r["onestep"] = float(((m(te[:, :-1]) - te[:, 1:]) ** 2).mean().sqrt())
        r.update(system=name, phase="test", seed=seed, method=tag)
        open(out, "a").write(json.dumps(r) + "\n"); print(r, flush=True)
