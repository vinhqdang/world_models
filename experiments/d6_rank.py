"""Model-level ranking-quality test of noise-coupling schemes (M=8 estimates vs an M_ref reference).

For S scenarios near the hazard (start-like states, goal, 32 candidate plans of varying lateral offset) it compares the
per-candidate expected-cost estimate from M=8 particles under each scheme to a large-M independent-noise reference:
  pearson / spearman over candidates, top-3 overlap, mean regret of the argmin pick, P(pick worse than best by > delta),
  model failure probability of the picked candidate.
It also runs a short CEM (N=32, iters=3) per scheme and scores the returned mean plan with the reference simulator
(`cem_*` columns): expected cost, model failure probability.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, LatentModel, FailureAnchor, observe
from selwm.risk_plan import cem
from selwm.d6_model import CoupledLatentModel, make_noise, SCHEMES

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='suite_hi/ckpt/es_s0.pt')
ap.add_argument('--schemes', default=','.join(SCHEMES))
ap.add_argument('--S', type=int, default=24); ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--Mref', type=int, default=2000); ap.add_argument('--R', type=int, default=40); ap.add_argument('--H', type=int, default=10)
ap.add_argument('--Scem', type=int, default=32); ap.add_argument('--Rcem', type=int, default=1); ap.add_argument('--Mcem_ref', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='results/d6/rank_s0.json')
ap.add_argument('--skip_cem', type=int, default=0); ap.add_argument('--cem_schemes', default='')
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = torch.device('cpu')

ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')); mem.load_state_dict(ck['state']); mem.eval()
variant = 'cliff_hi'
anchor = FailureAnchor(mem.encode, variant, mem.obs)
e = StochNav(variant, 1, 4242); sp, gp = e.sample_start_goal(512)
with torch.no_grad():
    zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
scale = float(((zs - zg_) ** 2).sum(-1).median())
kw = dict(stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
base = LatentModel(mem, variant, **kw)

rng = np.random.default_rng(args.seed + 100)


def scenarios(S):
    env = StochNav(variant, 1, int(rng.integers(1 << 30)))
    _, g = env.sample_start_goal(S)
    p = np.stack([rng.uniform(0.08, 0.80, S), rng.uniform(0.30, 0.50, S)], 1)
    return torch.tensor(p, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)


def candidates(p, g, N, H):
    S = p.shape[0]
    d = g - p
    d = d / d.norm(dim=-1, keepdim=True)
    yoff = torch.rand(S, N, 1) * 1.6 - 0.8                                   # lateral offset: negative = towards the pit
    eps = 0.4 * torch.randn(S, N, H, 2)
    act = d[:, None, None, :] + eps
    act[..., 1] = act[..., 1] + yoff
    act[:, 0] = d[:, None, :].expand(S, H, 2)                                # a plain straight candidate
    return act.clamp(-1, 1)


def ref_eval(p, g, acts, Mref, chunk=2):
    """Independent-noise large-M reference: (S,N) mean cost, mean failure flag."""
    S, N, H, _ = acts.shape
    cs, fs = [], []
    gen = torch.Generator().manual_seed(1234)
    for i in range(0, S, chunk):
        z0 = base.obs_to_latent(p[i:i + chunk]); zg = base.obs_to_latent(g[i:i + chunk])
        feat = base.rollout(z0, acts[i:i + chunk], Mref, gen, zg)
        c = base.cost(feat, zg)
        cs.append(c.mean(-1)); fs.append(feat[..., -1].mean(-1))
    return torch.cat(cs), torch.cat(fs)


def rank(x):
    return x.argsort(-1).argsort(-1).float()


def corr(x, y):
    x = x - x.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1) + 1e-12)


t0 = time.time()
p, g = scenarios(args.S)
acts = candidates(p, g, args.N, args.H)
refc, reff = ref_eval(p, g, acts, args.Mref)
print(f'reference done {time.time() - t0:.0f}s; ref cost range {refc.min():.3f}..{refc.max():.3f}, ref fail mean {reff.mean():.3f}', flush=True)
best_c = refc.min(-1).values
ref_top3 = refc.topk(3, -1, largest=False).indices
ref_top8 = refc.topk(8, -1, largest=False).indices
# reference noise floor: M=8 independent-sample estimate corr is the baseline; also report se of reference
z0 = base.obs_to_latent(p); zg = base.obs_to_latent(g)
results = {}
deltas = [0.1, 0.3]
for name in args.schemes.split(','):
    lm = CoupledLatentModel(mem, variant, noise=make_noise(name), **kw)
    gen = torch.Generator().manual_seed(args.seed * 1000 + 7)
    pe, sp_, t3, reg, fl, loc8 = [], [], [], [], [], []
    bad = {d: [] for d in deltas}
    for r in range(args.R):
        lm.begin_replan()
        c = lm.cost(lm.rollout(z0, acts, args.M, gen, zg), zg).mean(-1)                # (S,N)
        pe.append(corr(c, refc)); sp_.append(corr(rank(c), rank(refc)))
        sel = c.argmin(-1)
        rc = refc.gather(1, sel[:, None])[:, 0]
        reg.append(rc - best_c); fl.append(reff.gather(1, sel[:, None])[:, 0])
        for d in deltas:
            bad[d].append(((rc - best_c) > d).float())
        top3 = c.topk(3, -1, largest=False).indices
        t3.append(torch.tensor([len(set(top3[i].tolist()) & set(ref_top3[i].tolist())) / 3 for i in range(args.S)]))
        # rank correlation restricted to the 8 best candidates by reference
        loc8.append(corr(rank(c.gather(1, ref_top8)), rank(refc.gather(1, ref_top8))))
    m = lambda L: float(torch.stack(L).mean())
    results[name] = dict(pearson=m(pe), spearman=m(sp_), top3_overlap=m(t3), regret=m(reg), sel_fail=m(fl), spearman_top8=m(loc8),
                         **{f'p_bad_{d}': m(bad[d]) for d in deltas})
    print(name, {k: round(v, 4) for k, v in results[name].items()}, f'{time.time() - t0:.0f}s', flush=True)
results['_ref_best_fail'] = float(reff.gather(1, refc.argmin(-1)[:, None]).mean())

if not args.skip_cem:
    # CEM level: plan with each scheme (sticky state persists across the CEM iterations), score the plan under the reference
    pc, gc = scenarios(args.Scem)
    z0c = base.obs_to_latent(pc); zgc = base.obs_to_latent(gc)
    for name in (args.cem_schemes or args.schemes).split(','):
        results.setdefault(name, {})
        lm = CoupledLatentModel(mem, variant, noise=make_noise(name), **kw)
        gen = torch.Generator().manual_seed(args.seed * 1000 + 11)
        cc, ff = [], []
        for r in range(args.Rcem):
            lm.begin_replan()
            plan = cem(lm, z0c, zgc, args.H, args.N, args.M, iters=3, gen=gen)               # (S,H,2)
            c, f = ref_eval(pc, gc, plan[:, None], args.Mcem_ref, chunk=8)
            cc.append(c[:, 0]); ff.append(f[:, 0])
        results[name].update(cem_cost=float(torch.stack(cc).mean()), cem_fail=float(torch.stack(ff).mean()),
                             cem_cost_se=float(torch.stack(cc).mean(0).std() / np.sqrt(args.Scem)))
        print(name, 'CEM', {k: round(results[name][k], 4) for k in ('cem_cost', 'cem_fail', 'cem_cost_se')}, f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(args=vars(args), results=results), open(args.out, 'w'), indent=1)
