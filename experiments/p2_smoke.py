import sys, time, torch
sys.path.insert(0, '/home/user/world_models')
torch.set_num_threads(1)
from selwm.p2_dyn import *
from selwm.p2_train import *
name = sys.argv[1]
d = Data(name, seed=100, role="val")
print("data", d.train.shape, d.test.shape, d.ref.shape, float(d.train.std()))
for mode, kw in [("tf", {}), ("iso", dict(sigma=0.05)), ("ss", {}), ("lml", {})]:
    t = time.time(); m = fit(d, mode, seed=0, steps=1500, **kw)
    print(mode, round(time.time() - t, 1), "s", evaluate(m, d))
