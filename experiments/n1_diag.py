"""N1 mechanism diagnostic (no episodes): plan once per state with each arm at equal predictor-row budget, then evaluate the
returned plan on 1024 fresh independent noise draws under the same model (oracle = exact true simulator).
Reports, per arm: true (model) cost J of the returned plan, its failure probability, and optimism = J_fresh - in-sample score.
State seeds are >= 900 (disjoint from every evaluation seed).  usage: n1_diag.py --model oracle|learned --ms 0 --arms a,b,c
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from n1_common import *
from selwm.stochnav import StochNav

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='oracle'); ap.add_argument('--ms', type=int, default=0)
ap.add_argument('--arms', default='ol_indep,ol_crn3,cf_crn3,cf_indep')
ap.add_argument('--S', type=int, default=96); ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--Meval', type=int, default=1024); ap.add_argument('--seed', type=int, default=900)
ap.add_argument('--rho', type=float, default=0.0); ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_num_threads(1)
variant = 'cliff_hi'
rng = np.random.default_rng(args.seed)
e = StochNav(variant, 1, 0); e.rng = rng
s0, g0 = e.sample_start_goal(args.S)
near = rng.random(args.S) < 0.5                                       # half of the states: mid-path, close to the pit edge
s0[near, 0] = rng.uniform(0.2, 0.75, near.sum()); s0[near, 1] = rng.uniform(0.31, 0.46, near.sum())
P = torch.tensor(s0, dtype=torch.float32); G = torch.tensor(g0, dtype=torch.float32); F = torch.zeros(args.S, dtype=torch.bool)
ckpt = f'suite_hi/ckpt/es_s{args.ms}.pt' if args.model == 'learned' else None

def eval_plan(pl, plan, gen):
    """J, failure prob of plan (S,H,A) on Meval fresh iid draws."""
    z0, zg = pl.latent(P, F, G)
    out_J, out_f = [], []
    for ch in range(0, args.Meval, 256):
        u = torch.randn(args.S, 1, 256, pl.H, pl.K, generator=gen)
        feat = pl.lm.rollout_u(z0, plan[:, None], u, zg)
        c = pl.lm.cost(feat, zg)[:, 0]
        fl = (feat[:, 0, :, -1] if pl.model_kind == 'learned' else feat[:, 0, :, 2])
        out_J.append(c); out_f.append(fl)
    return torch.cat(out_J, 1).mean(1), torch.cat(out_f, 1).mean(1)

res = dict(args=vars(args), arms={})
for arm in args.arms.split(','):
    check_budget(arm)
    t0 = time.time(); Js, Fs, Os, Ch, Ss, True_opt, Meter = [], [], [], [], [], [], []
    for r in range(args.reps):
        pl = Planner(arm, args.model, ckpt, overrides=dict(rho=args.rho))
        gen = torch.Generator().manual_seed(1000 * args.seed + 17 * r + 1)
        pl.begin_replan(np.ones(args.S, bool))
        plan, d = pl.plan(P, F, G, gen, diag=True)
        J, f = eval_plan(pl, plan, torch.Generator().manual_seed(555 + r))
        Js.append(J); Fs.append(f)
        sA = d['sA_best']
        Jel, _ = eval_plan(pl, d['elite0'], torch.Generator().manual_seed(777 + r))
        True_opt.append(Jel - sA)
        if 'sB_of_Abest' in d: Meter.append(d['sB_of_Abest'] - sA)
        Ss.append(sA); Os.append(J - sA)
        if 'choice' in d: Ch.append(d['choice'].float())
        rows = pl.cnt['rows']
    Jm, Fm = torch.stack(Js).mean(0), torch.stack(Fs).mean(0)
    res['arms'][arm] = dict(J=Jm.tolist(), pfail=Fm.tolist(), J_mean=float(Jm.mean()), pfail_mean=float(Fm.mean()),
                            optimism_vs_Abest=float(torch.stack(Os).mean()), sA_best_mean=float(torch.stack(Ss).mean()),
                            optimism_true_Abest=float(torch.stack(True_opt).mean()), optimism_meter=(float(torch.stack(Meter).mean()) if Meter else None), optimism_true_per_state=torch.stack(True_opt).mean(0).tolist(), optimism_meter_per_state=(torch.stack(Meter).mean(0).tolist() if Meter else None), frac_choice_mean_plan=(float((torch.stack(Ch) == 0).float().mean()) if Ch else None),
                            sec=time.time() - t0)
    print(arm, {k: v for k, v in res['arms'][arm].items() if k not in ('J', 'pfail') and 'per_state' not in k}, flush=True)
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
