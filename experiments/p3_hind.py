"""P3 idea 4 (hindsight plan-ranking test): does the planner's own scoring prefer the logged action sequence that actually reached
a hindsight goal over other logged action sequences applied from the same start?

For each logged window (start p0, actions a_{0:H-1}, realised endpoint p_H, not fallen) the hindsight goal is g = encode(p_H).
The model-predicted planner cost J_m(p0, a, g) is computed for the logged plan and for K=63 plans taken from other logged windows.
Statistic: mean normalised rank of the logged plan (0 = best, 1 = worst) and the top-1 rate. No outcome labels are needed beyond endpoints.
"""
import argparse, json, os, sys, glob
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm import p3_offline as P3

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--name', required=True)
ap.add_argument('--n', type=int, default=1500); ap.add_argument('--K', type=int, default=63); ap.add_argument('--M', type=int, default=16)
args = ap.parse_args()
d = np.load('results/p3/val_cache.npz', allow_pickle=True)
W = {k: d['W_' + k] for k in ['p0', 'acts', 'traj', 'fell']}
m, lm, a = P3.load_model(args.ckpt)
rng = np.random.default_rng(31337); gen = torch.Generator().manual_seed(31337)
ok = np.where(~W['fell'].any(1))[0]
zone = ok[P3.zone_mask(W['p0'][ok])]
idx = rng.choice(zone, size=args.n, replace=False)
alt = rng.integers(0, len(W['p0']), (args.n, args.K))
t = lambda x: torch.as_tensor(x, dtype=torch.float32)
p0 = t(W['p0'][idx]); acts_log = t(W['acts'][idx]); pend = t(W['traj'][idx][:, -1])
z0 = P3.enc(m, p0); zg = P3.enc(m, pend)
acts_all = torch.cat([acts_log[:, None], t(W['acts'][alt])], 1)                 # (n, K+1, H, 2); index 0 = logged plan
n, K1 = acts_all.shape[:2]
with torch.no_grad():
    feat = lm.rollout(z0, acts_all, args.M, gen, zg)                           # (n,K+1,M,D+2)
    J = lm.cost(feat, zg).mean(-1)                                             # (n,K+1)
rank = (J[:, 1:] < J[:, :1]).float().mean(1)                                    # fraction of alternatives scored better than the logged plan
res = dict(name=args.name, hind_rank=float(rank.mean()), hind_top1=float((rank == 0).float().mean()), hind_top10=float((rank <= 0.1).float().mean()),
           hind_logJ=float(J[:, 0].mean()))
os.makedirs('results/p3/stats2', exist_ok=True)
json.dump(res, open(f'results/p3/stats2/{args.name}.json', 'w')); print(res)
