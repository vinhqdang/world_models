"""Offline diagnostic (learned model only): quality and planning-time optimism of the plan returned by each of the four arms.
For E start/goal pairs, plan once per arm (N=32, M, H=10, iters=3), then evaluate the returned plan (mean actions, mean gains) with
 (a) the noise used while planning (shared set of the last iteration, CRN arms only) and
 (b) fresh independent noise, M_eval = 256 particles (unbiased cost / failure estimate under the learned model).
optimism = (b) - (a) for the same plan.  Costs are the planner's cost (model.cost, includes the failure-anchor penalty)."""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.risk_plan import cem
from selwm.d3_fb import fb_cem
from selwm.d6_model import make_noise
from selwm.d9_model import CoupledLatentFB
from selwm.noise_aware import NoiseAwareScorer

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='0,1,2'); ap.add_argument('--E', type=int, default=64); ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--M', type=int, default=8); ap.add_argument('--Meval', type=int, default=256); ap.add_argument('--out', default='results/d9/diag.json')
args = ap.parse_args()
torch.manual_seed(0)
H, N, iters, variant = 10, 32, 3, 'cliff_hi'
ARMS = [('ol_indep', 'none', 'indep'), ('ol_crn3', 'none', 'crn_chain3'), ('fbstep_indep', 'step', 'indep'), ('fbstep_crn3', 'step', 'crn_chain3')]
res = {}
for ms in [int(x) for x in args.models.split(',')]:
    ck = torch.load(f'suite_hi/ckpt/es_s{ms}.pt', map_location='cpu'); a = ck['args']
    mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')); mem.load_state_dict(ck['state']); mem.eval()
    e0 = StochNav(variant, 1, 4242); sp, gp = e0.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
    scale = float(((zs - zg_) ** 2).sum(-1).median())
    anchor = FailureAnchor(mem.encode, variant, mem.obs, device=torch.device('cpu'))
    for arm, fb, sch in ARMS:
        lm = CoupledLatentFB(mem, variant, noise=make_noise(sch), stochastic=True, stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
        fresh = make_noise('indep')
        env = StochNav(variant, args.E, 777); env.reset()      # same states for every arm
        gen = torch.Generator().manual_seed(1)
        rows = []
        for rep in range(args.reps):
            s, g = env.sample_start_goal(args.E)
            p = torch.tensor(s, dtype=torch.float32); goal = torch.tensor(g, dtype=torch.float32)
            z0 = lm.obs_to_latent(p, torch.zeros(args.E, dtype=torch.bool)); zg = lm.obs_to_latent(goal)
            lm.begin_replan()
            sc = NoiseAwareScorer(0.0, 0.1, False)
            if fb == 'none':
                plan = cem(lm, z0, zg, H, N, args.M, iters=iters, gen=gen, score_fn=sc); K = torch.zeros(args.E, 1, 2)
            else:
                plan, K = fb_cem(lm, z0, zg, H, N, args.M, iters=iters, gen=gen, score_fn=sc, kmode='step', kmax=7.5)
            acts, Kk = plan[:, None], K[:, None]
            # (a) planning-time noise (shared sets only; for indep arms: a fresh M-particle draw = what a planner would see)
            if sch == 'indep':
                lm.noise = make_noise('indep'); lm.noise.fresh_replan = True
            c_a = lm.cost(lm.rollout_fb(z0, acts, Kk, args.M, gen, zg), zg).mean(-1)[:, 0]    # (E,)  for crn: the retained shared set
            # (b) fresh independent noise
            lm.noise = fresh
            f = lm.rollout_fb(z0, acts, Kk, args.Meval, gen, zg)
            c_b = lm.cost(f, zg).mean(-1)[:, 0]
            pf = f[..., -1].mean(-1)[:, 0]
            rows.append(torch.stack([c_a, c_b, pf], -1).detach().numpy())
        r = np.concatenate(rows)
        res[f'{arm}_m{ms}'] = dict(cost_plan_noise=float(r[:, 0].mean()), cost_fresh=float(r[:, 1].mean()), optimism=float((r[:, 1] - r[:, 0]).mean()),
                                  optimism_se=float((r[:, 1] - r[:, 0]).std(ddof=1) / np.sqrt(len(r))), p_fail_fresh=float(r[:, 2].mean()), n=len(r))
        print(arm, ms, res[f'{arm}_m{ms}'], flush=True)
json.dump(res, open(args.out, 'w'), indent=1)
agg = {}
for k, v in res.items():
    agg.setdefault(k.rsplit('_m', 1)[0], []).append(v)
print('\nmean over models:')
for k, vs in agg.items():
    print(f"{k:14s} cost_plan_noise={np.mean([v['cost_plan_noise'] for v in vs]):.3f} cost_fresh={np.mean([v['cost_fresh'] for v in vs]):.3f} optimism={np.mean([v['optimism'] for v in vs]):+.3f} p_fail_fresh={np.mean([v['p_fail_fresh'] for v in vs]):.3f}")
