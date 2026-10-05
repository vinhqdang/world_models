"""N2: Walsh (ANOVA on the hypercube) analysis of the planner cost when the learned noise is read as a binary path eps in {-1,+1}^H
(u_t = c eps_t vbar, complement 0).  For candidates scored over all 2^H = 1024 paths:
  * energy of the cost (and of candidate differences, which is what ranking under shared nodes depends on) by Walsh order |S|;
  * predicted rmse of M=8 estimators from the alias coefficients A_S = mean_m chi_S(x_m) of a design:
        iid MC            E err^2 = Var / M                         (average alias weight 1/M for every S)
        randomised design E err^2 = sum_S g_S^2 A_S^2 (averaged over random column permutation: mean A^2 by order)
  * agreement between the binary-path expectation and the Gaussian-noise Monte-Carlo reference (is the 2-atom reading faithful?)
"""
import argparse, itertools, json, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe, LatentModel
from selwm.n2_model import active_direction, oa_design
import importlib.util
spec = importlib.util.spec_from_file_location('h', 'experiments/n2_rank_helpers.py'); Hh = importlib.util.module_from_spec(spec); spec.loader.exec_module(Hh)

ap = argparse.ArgumentParser()
ap.add_argument('--model', type=int, default=0); ap.add_argument('--S', type=int, default=8); ap.add_argument('--N', type=int, default=16)
ap.add_argument('--H', type=int, default=10); ap.add_argument('--seed', type=int, default=5000); ap.add_argument('--c', type=float, default=2.0)
ap.add_argument('--out', default='results/n2/walsh.json')
args = ap.parse_args()
torch.manual_seed(args.seed)
variant = 'cliff_hi'
ck = torch.load(f'suite_hi/ckpt/es_s{args.model}.pt', map_location='cpu'); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); mem.load_state_dict(ck['state']); mem.eval()
anchor = FailureAnchor(mem.encode, variant, mem.obs)
e = StochNav(variant, 1, 4242); sp, gp = e.sample_start_goal(512)
with torch.no_grad():
    zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
scale = float(((zs - zg_) ** 2).sum(-1).median())
kw = dict(stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
lm = LatentModel(mem, variant, **kw)
vbar = active_direction(mem)
H = args.H; P = 2 ** H
eps = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=H)))              # (P,H); index bits: first column most significant
rng = np.random.default_rng(args.seed + 77); rng_t = torch.Generator().manual_seed(args.seed + 77)
order_of = torch.tensor([bin(i).count('1') for i in range(P)])
# Walsh transform in the {-1,+1} convention: chi_S(eps) = prod_{t in S} eps_t; index S bitmask with first column MSB
def walsh(g):                                                                    # g (...,P) -> coefficients (...,P) = mean_eps g chi_S
    x = g.clone()
    h = 1
    while h < P:
        x = x.reshape(*g.shape[:-1], P // (2 * h), 2, h)
        x = torch.stack([x[..., 0, :] + x[..., 1, :], x[..., 0, :] - x[..., 1, :]], -2).reshape(*g.shape[:-1], P)
        h *= 2
    return x / P
# sanity: with this transform the character of S (bitmask) is the product of eps over set bits; eps=+1 <-> value index bit 1 (product order above: -1 first)
res = {}
t0 = time.time()
for regime in ('wide', 'local'):
    p, g = Hh.scenarios(args.S, rng, variant)
    acts = Hh.candidates(p, g, args.N, H, regime, rng_t)
    z0 = lm.obs_to_latent(p); zg = lm.obs_to_latent(g)
    G = torch.zeros(args.S, args.N, P)
    with torch.no_grad():
        for s in range(args.S):
            z = z0[s][None, None, :].expand(args.N, P, -1)
            acc = torch.zeros(args.N, P); fl = torch.zeros(args.N, P)
            for t in range(H):
                at = acts[s, :, t][:, None, :].expand(args.N, P, 2)
                u = (args.c * eps[:, t])[None, :, None] * vbar[None, None, :].expand(args.N, P, -1)
                z = mem.pred(z, at, u)[0]
                acc = acc + ((z - zg[s]) ** 2).sum(-1) / H
                fl = torch.maximum(fl, (((z - anchor.z) ** 2).sum(-1) < anchor.tau).float())
            G[s] = (((z - zg[s]) ** 2).sum(-1) + acc) / scale + 3.0 * fl
    # Walsh spectra; bit ordering: index i has bits b_0 (MSB)...b_{H-1}; eps_t = -1 if bit t = 0 else +1; chi_S (S bitmask over the same bits)
    # the transform above is the standard Hadamard transform whose characters are (-1)^{popcount(i & S)}; for our eps coding that equals chi_S(eps) up to sign (-1)^{|S|}
    W = walsh(G)                                                                 # (S,N,P)
    Wd = []
    for s in range(args.S):                                                      # candidate differences relevant for ranking: all pairs
        d = G[s][:, None, :] - G[s][None, :, :]
        iu = torch.triu_indices(args.N, args.N, 1)
        Wd.append(walsh(d[iu[0], iu[1]]))
    Wd = torch.cat(Wd)
    def energy_by_order(Wc):
        E = torch.zeros(H + 1)
        for k in range(1, H + 1):
            E[k] = (Wc[..., order_of == k] ** 2).sum(-1).mean()
        return E
    Eg, Ed = energy_by_order(W), energy_by_order(Wd)
    # alias coefficient by order for designs (columns randomly permuted -> average A_S^2 over S of size k)
    def alias_by_order(X):                                                       # X (M,H) +-1 ; returns mean A^2 per order
        M = X.shape[0]; out = torch.zeros(H + 1)
        for k in range(1, H + 1):
            vals = []
            for S in itertools.combinations(range(H), k):
                vals.append(float(X[:, list(S)].prod(1).mean()) ** 2)
            out[k] = np.mean(vals)
        return out
    Xoa = torch.tensor(oa_design(8, H, 0), dtype=torch.float32)
    A_oa = alias_by_order(Xoa)
    # iid design (exact expectation 1/M), anti-iid: odd orders vanish exactly, even orders ~ 2/M
    A_iid = torch.full((H + 1,), 1 / 8.0)
    A_anti = torch.tensor([0.0] + [0.0 if k % 2 == 1 else 2 / 8.0 for k in range(1, H + 1)])
    # lhs-anti on a binary map: balanced column + antithetic: order 1 exact, order>=2 ~ 2/M roughly (reported through empirical rank test instead)
    def pred_rmse(Ek, A):
        return float(torch.sqrt((Ek * A).sum()))
    # exact binary-path expectation vs Gaussian reference is checked in the rank/atoms experiments; here report spectra
    tot_g = Eg.sum(); tot_d = Ed.sum()
    res[regime] = dict(
        energy_g_frac=(Eg / tot_g).tolist(), energy_diff_frac=(Ed / tot_d).tolist(),
        sd_g_mean=float(torch.sqrt(tot_g)), sd_diff_mean=float(torch.sqrt(tot_d)),
        alias_oa_by_order=A_oa.tolist(),
        pred_rmse_diff=dict(iid=pred_rmse(Ed, A_iid), anti=pred_rmse(Ed, A_anti), oa=pred_rmse(Ed, A_oa)),
        pred_rmse_g=dict(iid=pred_rmse(Eg, A_iid), anti=pred_rmse(Eg, A_anti), oa=pred_rmse(Eg, A_oa)),
        mean_cost=float(G.mean()), fail_sd=None)
    print(regime, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in res[regime].items()}), f'{time.time() - t0:.0f}s', flush=True)
json.dump(res, open(args.out, 'w'), indent=1)
