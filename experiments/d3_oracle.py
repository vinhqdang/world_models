"""Oracle (true simulator) test: open-loop CEM vs feedback CEM, evaluated in the TRUE closed loop with replanning.
Same streaming protocol as eval_stream.py (episodes restart on finish, ep_len 120)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.risk_plan import cem, OracleModel
from selwm.d3_fb import fb_cem, OracleFB

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='ol', help='ol | fb')
ap.add_argument('--kmode', default='const'); ap.add_argument('--kmax', type=float, default=15.0)
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--E', type=int, default=64)
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--H', type=int, default=10); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--total_steps', type=int, default=400); ap.add_argument('--ep_len', type=int, default=120)
ap.add_argument('--stage_w', type=float, default=1.0); ap.add_argument('--fail_cost', type=float, default=1.5)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default=None)
args = ap.parse_args()

model = OracleFB(args.variant, stage_w=args.stage_w); model.FAIL_COST = args.fail_cost
env = StochNav(args.variant, args.E, 5000 + args.seed); env.reset()
gen = torch.Generator().manual_seed(args.seed)
t_ep = np.zeros(args.E, int)
st = dict(done=0, fall=0, success=0, timeout=0, steps=[])
Ks = []
t0 = time.time()
for t in range(args.total_steps):
    p = torch.tensor(env.p, dtype=torch.float32); g = torch.tensor(env.goal, dtype=torch.float32)
    if args.mode == 'ol':
        plan = cem(model, p, g, args.H, args.N, args.M, iters=args.iters, gen=gen)
    else:
        plan, K = fb_cem(model, p, g, args.H, args.N, args.M, iters=args.iters, gen=gen, kmode=args.kmode, kmax=args.kmax)
        Ks.append(K.mean((0, 1)).tolist())
    env.step(plan[:, 0].numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    if done.any():
        for i in np.where(done)[0]:
            st['done'] += 1
            if env.fell[i]: st['fall'] += 1
            elif env.reached[i]: st['success'] += 1; st['steps'].append(int(t_ep[i]))
            else: st['timeout'] += 1
        s, gg = env.sample_start_goal(int(done.sum()))
        env.p[done] = s; env.goal[done] = gg; env.fell[done] = False; env.reached[done] = False; t_ep[done] = 0
n = max(1, st['done'])
res = dict(args=vars(args), episodes=st['done'], success=st['success'] / n, fall=st['fall'] / n, timeout=st['timeout'] / n,
           median_steps=float(np.median(st['steps'])) if st['steps'] else None,
           K_mean=(np.mean(Ks, 0).tolist() if Ks else None), sec=time.time() - t0)
print(json.dumps({k: v for k, v in res.items() if k != 'args'}), flush=True)
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
