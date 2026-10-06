"""P3: compute offline validation statistics for every candidate checkpoint (no ground-truth closed-loop information used).

python experiments/p3_stats.py --ckpt results/p3/ckpt/es_k500_s10.pt --name es_k500_s10
The logged validation data come from fixed seeds (777 one-step, 778 ordered windows) that no model was trained on.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm.stochnav import collect
from selwm import p3_offline as P3

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--out', default='results/p3/stats')
ap.add_argument('--skip_imag', type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
cache = 'results/p3/val_cache.npz'
if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    V = {k: d['V_' + k] for k in ['P', 'F', 'A', 'P2', 'F2']}
    W = {k: d['W_' + k] for k in ['p0', 'acts', 'traj', 'fell']}
else:
    P, F, A, P2, F2 = collect(P3.VARIANT, 100_000, 500, seed=777, edge_frac=0.25)
    V = dict(P=P, F=F, A=A, P2=P2, F2=F2)
    W = P3.make_windows(P3.collect_ordered(778, n_envs=300, T=400))
    np.savez(cache, **{'V_' + k: v for k, v in V.items()}, **{'W_' + k: v for k, v in W.items()})

m, lm, a = P3.load_model(args.ckpt)
res = dict(name=args.name, kind=a['kind'], steps=a['steps'], n_data=a['n_data'], edge=a['edge'], seed=a['seed'])

# ---------- baselines: held-out one-step predictive scores
os_ = P3.onestep_stats(m, lm, V)
res['ES1'] = float(os_['ES'].mean()); res['MSE1'] = float(os_['SE'].mean())
if m.kind != 'det':
    res['NLL1'] = float(os_['NLL'].mean())
else:                                                          # homoscedastic Gaussian NLL, sigma^2 fitted on the first half of the log
    n2 = len(os_['SE']) // 2
    s2 = max(os_['SE'][:n2].mean() / 3, 1e-6)
    res['NLL1'] = float((0.5 * (os_['SE'][n2:] / s2 + 3 * np.log(s2))).mean() * 1.0)
xy = 2 * V['P'] - 1
edge = (V['P'][:, 0] > 0.15) & (V['P'][:, 0] < 0.85) & (V['P'][:, 1] > 0.26) & (V['P'][:, 1] < 0.50)
res['ES1_edge'] = float(os_['ES'][edge].mean())            # plan-agnostic, domain-knowledge region near the pit
newfall = (V['F2'] & ~V['F']).astype(float)
res['Brier_hazard'] = float(((os_['HP'] - newfall) ** 2).mean())
res['Brier_hazard_edge'] = float(((os_['HP'] - newfall)[edge] ** 2).mean())
res.update(P3.multistep_stats(m, lm, W))

# ---------- planner-aware, plan-pool replay statistics (idea 1)
res.update({'pool_' + k: v for k, v in P3.plan_pool_stats(m, lm, W).items()})

# ---------- imagined closed loop (direct method) + occupancy-weighted validation (idea 2)
if not args.skip_imag:
    im, vis, vact = P3.imagined_closed_loop(m, lm)
    res.update(im)
    w, cid = P3.occupancy_weights(vis[:, :2], xy)
    wn = w[cid]
    res['OW_ES1'] = float((wn * os_['ES']).sum() / wn.sum())
    res['OW_Brier_hazard'] = float((wn * (os_['HP'] - newfall) ** 2).sum() / wn.sum())
    res['OW_ESS'] = float(wn.sum() ** 2 / (wn ** 2).sum() / len(wn))           # effective sample fraction of the weights
res['sec'] = time.time() - t0
json.dump(res, open(f'{args.out}/{args.name}.json', 'w'))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
