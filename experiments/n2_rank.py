"""N2 ranking-error benchmark: how well do M=8-equivalent estimators rank candidate plans against a high-budget reference?

Per model seed and candidate regime, S scenarios (start near the hazard, goal) x N=32 candidate plans.
Reference: scrambled-Sobol (H*K-dimensional) antithetic estimator with M_ref noise paths per candidate, independent noise.
Estimators (equal predictor evaluations, M=8 paths x H=10 steps = 80 evaluations per candidate; the atom models are counted):
  d6 baselines:  indep (iid per candidate), crn, crn_anti, crn_sobol_anti
  n2 designs:    dir_iid, dir_lhs, dir_lhs_anti, dir_oa (shared nodes along the model's active noise direction)
                 learned (offline-optimised shared nodes), atoms* / ut* (deterministic atom-cloud belief propagation)
Metrics per (scenario, replan-draw): Spearman, Pearson, rmse of centred cost, top-1 regret (reference cost of the picked plan
minus best), P(regret > delta), reference failure probability of the picked plan, top-3 overlap, mean reference cost of the 3
best-ranked ("elites").  Deterministic estimators are evaluated once.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, LatentModel, FailureAnchor, observe
from selwm.d6_model import CoupledLatentModel, make_noise
from selwm.n2_model import make_n2_model, AtomLatentModel

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='0,1,2'); ap.add_argument('--regimes', default='wide,local')
ap.add_argument('--S', type=int, default=30); ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--Mref', type=int, default=512); ap.add_argument('--R', type=int, default=30); ap.add_argument('--H', type=int, default=10)
ap.add_argument('--schemes', default='indep,crn,crn_anti,crn_sobol_anti,dir_iid,dir_lhs,dir_lhs_anti,dir_oa,atoms5,atoms5_fit,ut3,atoms11_fit')
ap.add_argument('--seed', type=int, default=5000); ap.add_argument('--learned_path', default='results/n2/learned_nodes_q.pt')
ap.add_argument('--noise_floor', type=int, default=1)
ap.add_argument('--out', default='results/n2/rank.json')
args = ap.parse_args()
torch.manual_seed(args.seed)
variant = 'cliff_hi'
deltas = [0.1, 0.3]


def rank(x):
    return x.argsort(-1).argsort(-1).float()


def corr(x, y):
    x = x - x.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1) + 1e-12)


def scenarios(S, rng):
    env = StochNav(variant, 1, int(rng.integers(1 << 30)))
    _, g = env.sample_start_goal(S)
    p = np.stack([rng.uniform(0.08, 0.80, S), rng.uniform(0.30, 0.50, S)], 1)
    return torch.tensor(p, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)


def candidates(p, g, N, H, regime, rng_t):
    S = p.shape[0]
    d = g - p; d = d / d.norm(dim=-1, keepdim=True)
    if regime == 'wide':                                               # d6 regime: large lateral spread, easy global ordering
        yoff = (torch.rand(S, N, 1, generator=rng_t) * 1.6 - 0.8)
        eps = 0.4 * torch.randn(S, N, H, 2, generator=rng_t)
    else:                                                              # late-CEM regime: local set around a plan, hard local ordering
        yoff = (torch.rand(S, 1, 1, generator=rng_t) * 0.8 - 0.1) + 0.12 * torch.randn(S, N, 1, generator=rng_t)
        eps = 0.15 * torch.randn(S, N, H, 2, generator=rng_t)
    act = d[:, None, None, :] + eps
    act[..., 1] = act[..., 1] + yoff
    return act.clamp(-1, 1)


class Counter:
    def __init__(self, pred):
        self.rows = 0
        pred.register_forward_hook(lambda mod, inp, out: setattr(self, 'rows', self.rows + inp[0].reshape(-1, inp[0].shape[-1]).shape[0]))


results = {}
t0 = time.time()
for mseed in [int(x) for x in args.models.split(',')]:
    ck = torch.load(f'suite_hi/ckpt/es_s{mseed}.pt', map_location='cpu'); a = ck['args']
    mem = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); mem.load_state_dict(ck['state']); mem.eval()
    anchor = FailureAnchor(mem.encode, variant, mem.obs)
    e = StochNav(variant, 1, 4242); sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
    scale = float(((zs - zg_) ** 2).sum(-1).median())
    kw = dict(stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
    counter = Counter(mem.pred)
    from selwm.d6_noise import NoiseSource
    ref_noise = NoiseSource(gen_kind='sobol80', anti=True, share=False)
    ref_model = CoupledLatentModel(mem, variant, noise=ref_noise, **kw)
    for regime in args.regimes.split(','):
        rng = np.random.default_rng(args.seed + 100 * mseed + (7 if regime == 'local' else 0))
        rng_t = torch.Generator().manual_seed(args.seed + 100 * mseed + (7 if regime == 'local' else 0))
        p, g = scenarios(args.S, rng)
        acts = candidates(p, g, args.N, args.H, regime, rng_t)
        z0 = ref_model.obs_to_latent(p); zg = ref_model.obs_to_latent(g)

        def ref_eval(seed):
            gen = torch.Generator().manual_seed(seed)
            cs, fs = [], []
            for i in range(0, args.S, 2):
                feat = ref_model.rollout(z0[i:i + 2], acts[i:i + 2], args.Mref, gen, zg[i:i + 2])
                cs.append(ref_model.cost(feat, zg[i:i + 2]).mean(-1)); fs.append(feat[..., -1].mean(-1))
            return torch.cat(cs), torch.cat(fs)
        refc, reff = ref_eval(1234)
        key0 = f'm{mseed}_{regime}'
        results[key0] = {'_ref_time': time.time() - t0}
        if args.noise_floor and mseed == 0:
            refc2, reff2 = ref_eval(98765)
            results[key0]['_noise_floor'] = dict(spearman=float(corr(rank(refc), rank(refc2)).mean()), rmse_centred=float(((refc - refc.mean(-1, keepdim=True)) - (refc2 - refc2.mean(-1, keepdim=True))).pow(2).mean().sqrt()),
                                                 ref_cost_sd_across_cands=float(refc.std(-1).mean()), fail_mean=float(reff.mean()), fail_sd_across_cands=float(reff.std(-1).mean()))
            print(key0, 'noise floor', results[key0]['_noise_floor'], f'{time.time() - t0:.0f}s', flush=True)
        best_c = refc.min(-1).values
        ref_top3 = refc.topk(3, -1, largest=False).indices
        for name in args.schemes.split(','):
            lm = make_n2_model(name, mem, variant, learned_path=args.learned_path, **kw)
            if lm is None:
                lm = CoupledLatentModel(mem, variant, noise=make_noise(name), **kw)
            deterministic = isinstance(lm, AtomLatentModel) or name == 'learned'
            R = 1 if deterministic else args.R
            gen = torch.Generator().manual_seed(args.seed + 31 * mseed + 5)
            met = {k: [] for k in ['spearman', 'pearson', 'rmse', 'regret', 'sel_fail', 'top3', 'elite_cost'] + [f'bad{d}' for d in deltas]}
            counter.rows = 0
            for r in range(R):
                if hasattr(lm, 'begin_replan'): lm.begin_replan()
                elif hasattr(lm, 'noise'): lm.noise.begin_replan()
                c = lm.cost(lm.rollout(z0, acts, args.M, gen, zg), zg).mean(-1)                 # (S,N)
                met['spearman'].append(corr(rank(c), rank(refc))); met['pearson'].append(corr(c, refc))
                met['rmse'].append(((c - c.mean(-1, keepdim=True)) - (refc - refc.mean(-1, keepdim=True))).pow(2).mean(-1).sqrt())
                sel = c.argmin(-1); rc = refc.gather(1, sel[:, None])[:, 0]
                met['regret'].append(rc - best_c); met['sel_fail'].append(reff.gather(1, sel[:, None])[:, 0])
                for d in deltas: met[f'bad{d}'].append(((rc - best_c) > d).float())
                top3 = c.topk(3, -1, largest=False).indices
                met['top3'].append(torch.tensor([len(set(top3[i].tolist()) & set(ref_top3[i].tolist())) / 3 for i in range(args.S)]))
                met['elite_cost'].append(refc.gather(1, top3).mean(-1))
            evals_per_cand = counter.rows / (R * args.S * args.N)
            per_scen = {k: torch.stack(v).mean(0).tolist() for k, v in met.items()}               # average over redraws -> (S,)
            results[key0][name] = dict(per_scen=per_scen, mean={k: float(np.mean(v)) for k, v in per_scen.items()}, evals_per_cand=float(evals_per_cand), deterministic=deterministic)
            print(key0, name, {k: round(v, 3) for k, v in results[key0][name]['mean'].items()}, f'evals/cand {evals_per_cand:.0f}', f'{time.time() - t0:.0f}s', flush=True)
        results[key0]['_ref_best_fail'] = float(reff.gather(1, refc.argmin(-1)[:, None]).mean())
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(dict(args=vars(args), results=results), open(args.out, 'w'))
