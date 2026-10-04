import sys, torch, numpy as np
sys.path.insert(0, '.')
from selwm.synth import SynthWorld
from selwm.pressure import cem_trajectory
torch.set_num_threads(4)

def run(**kw):
    w = SynthWorld(**kw)
    gen = torch.Generator().manual_seed(1)
    B, H, A, N, ITERS = 300, 5, 4, 200, 12
    a_star = w.new_contexts(B, gen)
    mus, kls = cem_trajectory(lambda acts: w.Jhat(acts.flatten(2), a_star), B, H, A, N, ITERS, gen=gen, init_std=1.0)
    mu_f = mus.flatten(2)                                   # (K+1,B,D)
    chat = torch.stack([w.Jhat(m[:, None], a_star)[:, 0] for m in mu_f]).numpy()
    J = torch.stack([w.J(m[:, None], a_star)[:, 0] for m in mu_f]).numpy()
    print(kw)
    print(' it   KL     c_hat    J_true   optimism(J-c_hat)')
    for k in range(ITERS + 1):
        print(f'{k:3d} {kls[k].mean():6.1f} {chat[k].mean():8.4f} {J[k].mean():8.4f} {(J[k]-chat[k]).mean():9.4f}')

for kw in [dict(s1=0.15, ell=1.0), dict(s1=0.5, ell=1.0), dict(s1=0.5, ell=0.5)]:
    run(**kw)
