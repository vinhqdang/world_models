"""D8: oracle (true simulator) open-loop vs feedback CEM; copy of experiments/d3_oracle.py with slot-wise common random numbers,
per-episode records, optional CVaR-mix scoring (--lam). Seeds >= 100 (D3 used 0-4)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.risk_plan import cem, OracleModel
from selwm.noise_aware import NoiseAwareScorer
from selwm.d3_fb import fb_cem, OracleFB

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='ol'); ap.add_argument('--kmode', default='step'); ap.add_argument('--kmax', type=float, default=7.5)
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--E', type=int, default=100)
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--H', type=int, default=10); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--total_steps', type=int, default=500); ap.add_argument('--ep_len', type=int, default=120)
ap.add_argument('--stage_w', type=float, default=1.0); ap.add_argument('--fail_cost', type=float, default=1.5)
ap.add_argument('--lam', type=float, default=0.0); ap.add_argument('--tail', type=float, default=0.1)
ap.add_argument('--seed', type=int, default=100); ap.add_argument('--out', required=True)
args = ap.parse_args()
model = OracleFB(args.variant, stage_w=args.stage_w); model.FAIL_COST = args.fail_cost
E = args.E
env = StochNav(args.variant, E, 0); env.rng = np.random.default_rng([7000, args.seed])
slot_env = [StochNav(args.variant, 1, 0) for _ in range(E)]
for i, se in enumerate(slot_env): se.rng = np.random.default_rng([8000, args.seed, i])
def draw(i):
    s, g = slot_env[i].sample_start_goal(1); return s[0], g[0]
S0, G0 = zip(*[draw(i) for i in range(E)]); env.reset(np.array(S0), np.array(G0))
ep_idx = np.zeros(E, int); gen = torch.Generator().manual_seed(args.seed); t_ep = np.zeros(E, int)
eps = []; Ks = []; t0 = time.time()
sf = NoiseAwareScorer(args.lam, args.tail, False) if args.lam > 0 else None
for t in range(args.total_steps):
    p = torch.tensor(env.p, dtype=torch.float32); g = torch.tensor(env.goal, dtype=torch.float32)
    if args.mode == 'ol':
        plan = cem(model, p, g, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=sf)
    else:
        plan, K = fb_cem(model, p, g, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=sf, kmode=args.kmode, kmax=args.kmax)
        Ks.append(K.mean((0, 1)).tolist())
    env.step(plan[:, 0].numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    if done.any():
        for i in np.where(done)[0]:
            o = 'fall' if env.fell[i] else ('success' if env.reached[i] else 'timeout')
            eps.append((int(i), int(ep_idx[i]), o, int(t_ep[i]))); ep_idx[i] += 1
            s, gg = draw(i); env.p[i] = s; env.goal[i] = gg; env.fell[i] = False; env.reached[i] = False; t_ep[i] = 0
n = len(eps); cnt = {o: sum(1 for e in eps if e[2] == o) for o in ('success', 'fall', 'timeout')}
ps = args.iters * args.N * (args.M + (1 if args.mode == 'fb' else 0)) * args.H
res = dict(args=vars(args), episodes=n, success=cnt['success'] / n, fall=cnt['fall'] / n, timeout=cnt['timeout'] / n,
           particle_steps_per_env_step=ps, K_mean=(np.mean(Ks, 0).tolist() if Ks else None), sec=time.time() - t0, records=eps)
print(json.dumps({k: v for k, v in res.items() if k not in ('args', 'records')}))
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
