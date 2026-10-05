"""Closed-loop planning evaluation of trained JEPA world models on the stochastic navigation benchmark.

python experiments/eval_plan.py --ckpt runs/es_s0.pt --planner expected
planner: mean | expected | cvar_<alpha>   (mean uses the deterministic predictor path, others sample particles)
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav, render
from selwm.jepa import JEPA, LatentModel
from selwm.risk_plan import cem

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--planner', default='expected')
ap.add_argument('--variant', default='cliff'); ap.add_argument('--E', type=int, default=100)
ap.add_argument('--N', type=int, default=128); ap.add_argument('--M', type=int, default=16)
ap.add_argument('--H', type=int, default=12); ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--crn', type=int, default=0)
ap.add_argument('--max_steps', type=int, default=120); ap.add_argument('--out', default=None)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
dev = torch.device(args.device)

ck = torch.load(args.ckpt, map_location=dev)
a = ck['args']
jepa = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')).to(dev)
jepa.load_state_dict(ck['state']); jepa.eval()
stochastic = args.planner != 'mean'
lm = LatentModel(jepa, args.variant, stochastic=stochastic, crn=bool(args.crn))
risk = 'expected' if args.planner == 'mean' else args.planner.split('_')[0]
alpha = float(args.planner.split('_')[1]) if '_' in args.planner else 0.25

res = []
for seed in range(args.seeds):
    env = StochNav(args.variant, args.E, 1000 + seed); env.reset()
    gen = torch.Generator(device=dev).manual_seed(seed)
    zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32, device=dev))
    steps_taken = np.zeros(args.E)
    t0 = time.time()
    for t in range(args.max_steps):
        p = torch.tensor(env.p, dtype=torch.float32, device=dev)
        fell = torch.tensor(env.fell, device=dev)
        z0 = lm.obs_to_latent(p, fell)
        plan = cem(lm, z0, zg, args.H, args.N, args.M, iters=args.iters, risk=risk, alpha=alpha, gen=gen)
        env.step(plan[:, 0].cpu().numpy())
        alive = ~(env.fell | env.reached)
        steps_taken += alive
        if not alive.any():
            break
    res.append(dict(seed=seed, success=float(env.reached.mean()), fall=float(env.fell.mean()),
                    timeout=float((~env.reached & ~env.fell).mean()), steps=float(steps_taken[env.reached].mean() if env.reached.any() else np.nan),
                    sec=time.time() - t0))
    print(args.planner, os.path.basename(args.ckpt), res[-1], flush=True)
s = np.array([r['success'] for r in res]); f = np.array([r['fall'] for r in res])
print(f'SUMMARY {os.path.basename(args.ckpt)} {args.planner}: success {s.mean():.3f}+/-{s.std(ddof=1)/max(1,len(s))**.5:.3f} fall {f.mean():.3f}')
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(dict(args=vars(args), runs=res), open(args.out, 'w'))
