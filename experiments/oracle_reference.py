"""Reference planner that uses the TRUE simulator as its model (upper bound for the learned-model planners).

Streaming closed-loop evaluation on cliff_hi, same episode protocol as eval_stream.py (episodes restart on finish, ep_len 120,
goal radius 0.06). Two compute settings: a large one and the one matched to the learned-model experiments (N=32, M=8, H=10).
python experiments/oracle_reference.py --wind 0.09 --N 64 --M 12 --H 12 --out results/oracle/exp_w09_N64.json
"""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.risk_plan import cem, OracleModel

ap = argparse.ArgumentParser()
ap.add_argument('--wind', type=float, default=0.09); ap.add_argument('--planner', default='expected')
ap.add_argument('--N', type=int, default=64); ap.add_argument('--M', type=int, default=12); ap.add_argument('--H', type=int, default=12)
ap.add_argument('--iters', type=int, default=3); ap.add_argument('--E', type=int, default=64); ap.add_argument('--steps', type=int, default=450)
ap.add_argument('--seed', type=int, default=211); ap.add_argument('--out', required=True)
a = ap.parse_args()
torch.set_num_threads(1)
env = StochNav('cliff_hi', a.E, 5000 + a.seed, wind=a.wind); env.reset()
gen = torch.Generator().manual_seed(a.seed)
model = OracleModel('cliff_hi', deterministic=(a.planner == 'mean'), wind=a.wind, stage_w=1.0)
t_ep = np.zeros(a.E, int); warm = None
cnt = dict(success=0, fall=0, timeout=0); steps_goal = []
t0 = time.time()
for t in range(a.steps):
    p = torch.tensor(env.p, dtype=torch.float32); g = torch.tensor(env.goal, dtype=torch.float32)
    plan = cem(model, p, g, a.H, a.N, a.M, iters=a.iters, risk='expected', gen=gen, warm=warm)
    warm = torch.cat([plan[:, 1:], torch.zeros_like(plan[:, :1])], 1)
    env.step(plan[:, 0].numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= 120)
    for i in np.where(done)[0]:
        if env.fell[i]: cnt['fall'] += 1
        elif env.reached[i]: cnt['success'] += 1; steps_goal.append(int(t_ep[i]))
        else: cnt['timeout'] += 1
    if done.any():
        s, gg = env.sample_start_goal(int(done.sum()))
        env.p[done] = s; env.goal[done] = gg; env.fell[done] = False; env.reached[done] = False; t_ep[done] = 0; warm = None
n = sum(cnt.values())
res = dict(args=vars(a), episodes=n, success=cnt['success'] / n, fall=cnt['fall'] / n, timeout=cnt['timeout'] / n,
           median_steps=float(np.median(steps_goal)) if steps_goal else None, sec=time.time() - t0)
json.dump(res, open(a.out, 'w'))
print(json.dumps({k: v for k, v in res.items() if k != 'args'}))
