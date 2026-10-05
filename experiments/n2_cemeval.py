"""N2 CEM-level test (exploitation check): plan with CEM (N=32, M=8-equivalent, H=10, 3 iterations) under each estimator, then score the
returned mean plan with an independent high-budget reference (scrambled-Sobol antithetic, M_ref noise paths): reference expected cost
and failure probability.  This is where a fixed (deterministic) node set could be exploited by the optimiser; ranking metrics on random
candidates cannot detect that.  Start/goal states are drawn like the closed-loop benchmark (seeds 7000+, disjoint from all other seeds)."""
import argparse, json, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.risk_plan import cem
from selwm.noise_aware import NoiseAwareScorer
from selwm.d6_model import CoupledLatentModel, make_noise
from selwm.d6_noise import NoiseSource
from selwm.n2_model import make_n2_model, AtomLatentModel

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='0,1,2'); ap.add_argument('--S', type=int, default=64); ap.add_argument('--R', type=int, default=3)
ap.add_argument('--Mref', type=int, default=1024); ap.add_argument('--schemes', default='indep,crn_anti,crn_chain3,dir_oa,learned,atoms5_fit')
ap.add_argument('--seed', type=int, default=7000); ap.add_argument('--out', default='results/n2/cemeval.json')
ap.add_argument('--learned_path', default='results/n2/learned_nodes_q.pt')
args = ap.parse_args()
variant = 'cliff_hi'; H = 10; N = 32; M = 8
res = {}
t0 = time.time()
for ms in [int(x) for x in args.models.split(',')]:
    ck = torch.load(f'suite_hi/ckpt/es_s{ms}.pt', map_location='cpu', weights_only=False); a = ck['args']
    mem = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); mem.load_state_dict(ck['state']); mem.eval()
    anchor = FailureAnchor(mem.encode, variant, mem.obs)
    e = StochNav(variant, 1, 4242); sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
    scale = float(((zs - zg_) ** 2).sum(-1).median())
    kw = dict(stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
    env = StochNav(variant, 1, args.seed + ms)
    s0, g0 = env.sample_start_goal(args.S)
    # put start positions anywhere along the approach (x in 0.06..0.8) so that the hazard is in play
    rng = np.random.default_rng(args.seed + ms)
    s0[:, 0] = rng.uniform(0.06, 0.75, args.S)
    p = torch.tensor(s0, dtype=torch.float32); g = torch.tensor(g0, dtype=torch.float32)
    ref_model = CoupledLatentModel(mem, variant, noise=NoiseSource(gen_kind='sobol80', anti=True, share=False), **kw)
    z0 = ref_model.obs_to_latent(p); zg = ref_model.obs_to_latent(g)
    def ref(plan):                                             # plan (S,H,2) -> cost (S,), fail (S,)
        gen = torch.Generator().manual_seed(999)
        cs, fs = [], []
        for i in range(0, args.S, 8):
            feat = ref_model.rollout(z0[i:i + 8], plan[i:i + 8, None], args.Mref, gen, zg[i:i + 8])
            cs.append(ref_model.cost(feat, zg[i:i + 8]).mean(-1)[:, 0]); fs.append(feat[..., -1].mean(-1)[:, 0])
        return torch.cat(cs), torch.cat(fs)
    for name in args.schemes.split(','):
        lm = make_n2_model(name, mem, variant, learned_path=args.learned_path, **kw)
        if lm is None:
            lm = CoupledLatentModel(mem, variant, noise=make_noise(name), **kw)
        costs, fails = [], []
        for r in range(args.R):
            gen = torch.Generator().manual_seed(args.seed + 97 * r + ms)
            if hasattr(lm, 'begin_replan'): lm.begin_replan(reset_mask=np.ones(args.S, bool))
            elif hasattr(lm, 'noise'): lm.noise.begin_replan(np.ones(args.S, bool))
            plan = cem(lm, z0, zg, H, N, M, iters=3, gen=gen, score_fn=NoiseAwareScorer(0.0, 0.1, False))
            c, f = ref(plan); costs.append(c); fails.append(f)
        c = torch.stack(costs).mean(0); f = torch.stack(fails).mean(0)
        res.setdefault(f'm{ms}', {})[name] = dict(cost=c.tolist(), fail=f.tolist())
        print(ms, name, f'ref cost {c.mean():.3f}  ref fail {f.mean():.3f}  {time.time() - t0:.0f}s', flush=True)
json.dump(dict(args=vars(args), results=res), open(args.out, 'w'))
