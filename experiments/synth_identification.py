"""Synthetic identification + control experiment.

Every round the planner runs CEM for K iterations (all iterates and their model costs are known),
but only the TRUE cost of the single iterate it executes is revealed. Policies:
  last      : always the final iterate (default CEM)
  fixed_k   : best fixed iterate chosen in hindsight (oracle tuning)
  klpen     : minimise c_hat_k + lam * KL_k with lam oracle-tuned in hindsight (fixed-penalty baseline)
  online    : ours -- dithered exploration + isotonic estimate of the optimism curve G(KL)
  oracle_r  : per-round best iterate (not attainable; needs J)
"""
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from selwm.synth import SynthWorld
from selwm.pressure import cem_trajectory, OptimismCurve
torch.set_num_threads(4)

def make_rounds(world, R, H=5, A=4, N=200, ITERS=12, seed=1):
    gen = torch.Generator().manual_seed(seed)
    a_star = world.new_contexts(R, gen)
    mus, kls = cem_trajectory(lambda acts: world.Jhat(acts.flatten(2), a_star), R, H, A, N, ITERS, gen=gen, init_std=1.0)
    mu_f = mus.flatten(2)
    chat = torch.stack([world.Jhat(m[:, None], a_star)[:, 0] for m in mu_f]).numpy()
    J = torch.stack([world.J(m[:, None], a_star)[:, 0] for m in mu_f]).numpy()
    return kls.numpy(), chat, J                      # (K+1, R)

def online_policy(kls, chat, J, rng, warm=20, eps_floor=0.1):
    K1, R = kls.shape
    grid = np.arange(1, K1)
    cur = OptimismCurve(); cur.fit()
    chosen = np.zeros(R, int)
    for t in range(R):
        eps = max(eps_floor, 1.0 / np.sqrt(t + 1)) if t >= warm else 1.0
        if rng.random() < eps:
            k = rng.choice(grid)
        else:
            k = grid[np.argmin(chat[grid, t] + cur(kls[grid, t]))]
        chosen[t] = k
        cur.add(kls[k, t], J[k, t] - chat[k, t]); cur.fit()
    return chosen

out = []
for seed in range(10):
    world = SynthWorld(s1=0.5, ell=1.0, seed=seed)
    kls, chat, J = make_rounds(world, R=1500, seed=100 + seed)
    K1, R = J.shape
    rng = np.random.default_rng(seed)
    chosen = online_policy(kls, chat, J, rng)
    tail = slice(R // 2, R)                                    # compare on the second half (after learning)
    Jo = J[chosen, np.arange(R)]
    best_fixed = J.mean(1)[1:].min(); k_bf = J.mean(1)[1:].argmin() + 1
    lams = np.logspace(-4, 0, 25)
    klpen = min(J[np.argmin(chat[1:] + l * kls[1:], 0) + 1, np.arange(R)][tail].mean() for l in lams)
    row = dict(seed=seed, last=float(J[-1][tail].mean()), fixed_k=float(J[k_bf][tail].mean()), best_fixed_k=int(k_bf),
               klpen=float(klpen), online=float(Jo[tail].mean()), online_exec_tail=float(Jo[tail].mean()), oracle_r=float(J[1:].min(0)[tail].mean()),
               regret_last_pct=0.0)
    out.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

keys = ['last', 'fixed_k', 'klpen', 'online', 'oracle_r']
print('\nmean true cost of executed plans (2nd half of the stream), mean over 10 worlds +/- s.e.:')
for k in keys:
    v = np.array([r[k] for r in out]); print(f'  {k:9s} {v.mean():.4f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.4f}')
json.dump(out, open('results/synth_identification.json', 'w'), indent=1)
