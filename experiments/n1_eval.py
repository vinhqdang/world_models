"""N1: streaming closed-loop evaluation on cliff_hi (learned energy-score models or the true simulator) of open-loop CEM
versus cross-fitted CEM at equal predictor-row budget.  Harness layout copied from experiments/d8_eval.py (slot-wise start/goal
generators so that arms with the same --seed see identical start/goal for each (slot, episode index); wind stream per seed)
with the noise coupling of experiments/d9_eval.py (begin_replan with the episode-reset mask).
Evaluation seeds used for the reported results: 400, 401, 402.   (never used for tuning; diagnostics used seeds >= 900)
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from n1_common import *
from selwm.stochnav import StochNav

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='learned'); ap.add_argument('--ms', type=int, default=0)
ap.add_argument('--arm', required=True)
ap.add_argument('--E', type=int, default=16); ap.add_argument('--total_steps', type=int, default=500)
ap.add_argument('--ep_len', type=int, default=120); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_num_threads(1)
variant = 'cliff_hi'
check_budget(args.arm)
np.random.seed(args.seed); torch.manual_seed(args.seed)
ckpt = f'suite_hi/ckpt/es_s{args.ms}.pt' if args.model == 'learned' else None
pl = Planner(args.arm, args.model, ckpt, variant)
E = args.E
env = StochNav(variant, E, 0); env.rng = np.random.default_rng([7000, args.seed])
slot_env = [StochNav(variant, 1, 0) for _ in range(E)]
for i, se in enumerate(slot_env): se.rng = np.random.default_rng([8000, args.seed, i])
def draw(i):
    s, g = slot_env[i].sample_start_goal(1); return s[0], g[0]
S0, G0 = zip(*[draw(i) for i in range(E)]); env.reset(np.array(S0), np.array(G0))
ep_idx = np.zeros(E, int); t_ep = np.zeros(E, int)
gen = torch.Generator().manual_seed(args.seed)
prev_reset = np.ones(E, bool)
eps = []; opt_trace = []; choice_trace = []
t0 = time.time()
for t in range(args.total_steps):
    p = torch.tensor(env.p, dtype=torch.float32); fell = torch.tensor(env.fell); goal = torch.tensor(env.goal, dtype=torch.float32)
    pl.begin_replan(prev_reset)
    plan, d = pl.plan(p, fell, goal, gen, diag=True)
    if 'choice' in d:
        opt_trace.append(float((d['sB_of_Abest'] - d['sA_best']).mean())); choice_trace.append(float((d['choice'] == 0).float().mean()))
    env.step(plan[:, 0].numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    prev_reset = np.zeros(E, bool)
    if done.any():
        for i in np.where(done)[0]:
            o = 'fall' if env.fell[i] else ('success' if env.reached[i] else 'timeout')
            eps.append((int(i), int(ep_idx[i]), o, int(t_ep[i]))); ep_idx[i] += 1
            s, g = draw(i); env.p[i] = s; env.goal[i] = g; env.fell[i] = False; env.reached[i] = False; t_ep[i] = 0
            prev_reset[i] = True
    if (t + 1) % 100 == 0:
        print(f'step {t+1} eps={len(eps)} {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
n = len(eps); cnt = {o: sum(1 for e in eps if e[2] == o) for o in ('success', 'fall', 'timeout')}
stg = [e[3] for e in eps if e[2] == 'success']
res = dict(args=vars(args), episodes=n, success=cnt['success'] / max(1, n), fall=cnt['fall'] / max(1, n), timeout=cnt['timeout'] / max(1, n),
           counts=cnt, median_steps=float(np.median(stg)) if stg else None,
           rows_per_env_step=pl.cnt['rows'] / args.total_steps if args.model == 'learned' else None,
           optimism_B_minus_A_mean=(float(np.mean(opt_trace)) if opt_trace else None),
           frac_mean_plan_chosen=(float(np.mean(choice_trace)) if choice_trace else None),
           sec=time.time() - t0, records=eps)
print(json.dumps({k: v for k, v in res.items() if k not in ('args', 'records')}))
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
