"""Benchmark sanity: closed-loop MPC with the TRUE simulator as the model."""
import sys, time, json, numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.risk_plan import cem, OracleModel
torch.set_num_threads(4)

def evaluate(variant, planner, E=100, H=12, N=150, M=16, iters=4, seed=0, max_steps=70, **kw):
    env = StochNav(variant, E, seed)
    env.reset()
    gen = torch.Generator().manual_seed(seed)
    model = OracleModel(variant, deterministic=(planner == 'mean'))
    risk = 'expected' if planner == 'mean' else planner.split('_')[0]
    alpha = float(planner.split('_')[1]) if '_' in planner else 0.25
    for t in range(max_steps):
        p = torch.tensor(env.p, dtype=torch.float32); g = torch.tensor(env.goal, dtype=torch.float32)
        plan = cem(model, p, g, H, N, M, iters=iters, risk=risk, alpha=alpha, gen=gen, **kw)
        env.step(plan[:, 0].numpy())
        if (env.fell | env.reached).all(): break
    return env.reached.mean(), env.fell.mean(), t + 1

for variant in ['cliff']:
    for planner in ['mean', 'expected', 'cvar_0.5', 'cvar_0.25', 'cvar_0.1']:
        t0 = time.time(); rs, fs = [], []
        for seed in range(3):
            r, f, T = evaluate(variant, planner, seed=seed); rs.append(r); fs.append(f)
        print(f'{variant:6s} {planner:10s} success {np.mean(rs):.3f} +/- {np.std(rs, ddof=1)/np.sqrt(3):.3f}   fall {np.mean(fs):.3f}   ({time.time()-t0:.0f}s)', flush=True)
