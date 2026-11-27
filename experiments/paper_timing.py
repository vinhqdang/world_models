"""Timing of the heavy arms on synthetic arrays of the sizes of one fit seed (for the compute estimate in PROTOCOL.md).
usage: python3 experiments/paper_timing.py D R [THREADS]      (d outputs, R drivers; n = 70000 train, 7500 held out, 32768 test points)"""
import sys, time, json
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import eb as EB

d, R = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
rng = np.random.default_rng(0)
q = 1 + d
n, nv, nt = 70000, 7500, 32768
G0 = rng.standard_normal((q, d, R)) * 0.5
psi = lambda m: np.c_[np.ones(m), rng.uniform(-1, 1, (m, d))]
def make(m):
    P = psi(m); G = EB.G_of(P, G0)
    xi = (rng.random((m, R)) < 0.2) * rng.uniform(-1, 1, (m, R)) * 2.5
    r = torch.einsum("bdr,br->bd", G, torch.as_tensor(xi)) + 0.05 * torch.randn(m, d, dtype=torch.float64)
    return torch.as_tensor(P), G, r
Ptr, Gtr, rtr = make(n); Pva, Gva, rva = make(nv); Pt, Gt, rt = make(nt)
out = dict(d=d, R=R)
def tm(name, f):
    t = time.time(); r = f(); out[name] = round(time.time() - t, 1); print(name, out[name], "s", flush=True); return r
tm("gmm_one_fit_K32_3000", lambda: EB.fit_gmm(Gtr, rtr, Gva, rva, K=32, steps=3000, seed=0))
tm("gmm_one_fit_K16_3000", lambda: EB.fit_gmm(Gtr, rtr, Gva, rva, K=16, steps=3000, seed=0))
m = tm("flow_fit_1500", lambda: EB.fit_flow(Gtr, rtr, Gva, rva, steps=1500, seed=0))
tm("flow_posterior_4096_32768pts", lambda: EB.rc_flow_posterior(m[0], Gt, rt, S=4096))
tm("npmle_5noise_x150it", lambda: EB.rc_npmle(Gtr, rtr, Gva, rva, seed=0))
tm("idt_one_fit_3000", lambda: EB.fit_idt(torch.as_tensor(G0), Ptr, rtr, Pva, rva, K=4, steps=3000, restarts=1, seed=0))
print(json.dumps(out))
open(f"/home/user/world_models/results/paper/logs/timing_d{d}_R{R}.json", "w").write(json.dumps(out))
