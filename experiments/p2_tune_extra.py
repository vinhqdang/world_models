"""extra validation tuning for scheduled sampling at p=0.9 and for the longer vdp horizon (val seed 100 only)"""
import sys, json, torch
sys.path.insert(0, '/home/user/world_models'); torch.set_num_threads(1)
from selwm.p2_dyn import *; from selwm.p2_train import *
name = sys.argv[1]
d = Data(name, seed=100, n_train=100, role="val")
T = local_tangent_normal(d.train.reshape(-1, D_OBS), d.S["dim"], k=32)
for tag, kw in [("ss@0.9", dict(mode="ss", ss_p=0.9, H=8)), ("ss@0.6", dict(mode="ss", ss_p=0.6, H=8)),
                ("iso@0.03", dict(mode="iso", sigma=0.03)), ("normal@0.03", dict(mode="normal", sigma=0.03, tangents=T)),
                ("normal@0.05", dict(mode="normal", sigma=0.05, tangents=T)), ("iso@0.05", dict(mode="iso", sigma=0.05))]:
    m = fit(d, seed=0, steps=4000, bs=1024, lr=2e-3, **kw)
    r = evaluate(m, d); r.update(system=name, phase="tune2", seed=0, method=tag)
    open(f"/home/user/world_models/results/p2/{name}_tune2.jsonl", "a").write(json.dumps(r) + "\n"); print(r, flush=True)
