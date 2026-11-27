"""EXPLORATORY: noisy independent-factor head r = G(h,a) xi + n, xi_j ~ MoG independent (K comps), n ~ N(0, diag).
K=1 is the Gaussian factor model (gauge-arbitrary when G is free in a). Transport: G(a') E[xi | r, a] (closed form)."""
import sys, pickle, math, time, itertools
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch, torch.nn as nn
from cwk import kernels as KR, metrics as M
from cwk.predictor import mlp
torch.set_num_threads(2)
name = sys.argv[1]; mode = sys.argv[2]; Kc = int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]; steps = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
Z = pickle.load(open(f"{name}_s0.pkl", "rb"))
kd, hs, psi_pop, r_std, mu_pop, truth, coords = Z["kd"], Z["hs"], Z["psi_pop"], Z["r_std"], Z["mu_pop"], Z["truth"], Z["coords"]
N, E, K, D = truth.shape; d = len(coords); R = 2
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
base = np.repeat(mu_pop[:, None], E, 1)
def score(deltas):
    pred = base.copy()
    for k in range(1, K):
        pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + kd.sigma * deltas[k - 1].numpy().reshape(N, E, d)
    return round(100 * (1 - ev.loss(pred).mean() / ev.var), 2)
bases = {"lin": lambda p: p, "cub": lambda p: torch.cat([p, p[:, 1:] ** 3], -1), "quadcub": lambda p: torch.cat([p, p[:, 1:] ** 2, p[:, 1:] ** 3], -1)}
class Gmod(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        if mode in bases:
            q = bases[mode](kd.psi[:2]).shape[1]; self.Gj = nn.Parameter(0.3 * torch.randn(q, d, R))
        else:
            self.net = mlp((kd.dh if mode == "mlp_ha" else 0) + kd.q, d * R, 256, 3)
            with torch.no_grad(): self.net[-1].weight.mul_(0.3); self.net[-1].bias.normal_(0, 0.4)
    def forward(self, h, psi):
        if self.mode in bases: return torch.einsum("bq,qdr->bdr", bases[self.mode](psi), self.Gj)
        return self.net(torch.cat([h, psi], -1) if self.mode == "mlp_ha" else psi).view(-1, d, R)
combos = torch.tensor(list(itertools.product(range(Kc), repeat=R)))            # (C, R)
class Src(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.linspace(-1, 1, Kc).repeat(R, 1) if Kc > 1 else torch.zeros(R, 1)); self.ls = nn.Parameter(torch.full((R, Kc), -0.5 if Kc > 1 else 0.0)); self.lw = nn.Parameter(torch.zeros(R, Kc))
    def comps(self):
        ar = torch.arange(R); lw = torch.log_softmax(self.lw, -1)
        m = self.m if Kc > 1 else self.m * 0; s = torch.exp(2 * self.ls) if Kc > 1 else torch.ones(R, 1)
        return lw[ar, combos].sum(-1), m[ar, combos], s[ar, combos]              # (C,), (C,R), (C,R)
def loglik_post(G, r, src, logn):
    lw, m, s = src.comps(); G = G.double(); r = r.double(); n = (1e-3 + torch.exp(logn)).double(); m = m.double(); s = s.double()
    mean = torch.einsum("bdr,cr->bcd", G, m)                                         # (B,C,d)
    Gs = G[:, None] * s[None, :, None, :]                                            # (B,C,d,R)
    cov = Gs @ G[:, None].transpose(-1, -2) + torch.diag(n)                          # (B,C,d,d)
    L = torch.linalg.cholesky(cov); dev = (r[:, None] - mean)
    sol = torch.cholesky_solve(dev[..., None], L)[..., 0]
    lp = lw.double()[None] - 0.5 * (dev * sol).sum(-1) - torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    ll = torch.logsumexp(lp, 1)
    post = torch.softmax(lp, 1)                                                      # (B,C)
    xi_c = m[None] + torch.einsum("bcdr,bcd->bcr", Gs, sol)                          # E[xi | r, c]
    return ll, (post[..., None] * xi_c).sum(1)
rng0 = np.random.default_rng(0); eps_ids = np.unique(kd.ep); val = set(rng0.choice(eps_ids, len(eps_ids) // 10, replace=False).tolist())
isv = np.array([e in val for e in kd.ep]); tr, va = np.where(~isv)[0], np.where(isv)[0]
for sd in seeds:
    t0 = time.time(); torch.manual_seed(sd); rng = np.random.default_rng(sd)
    g, src, logn = Gmod(mode), Src(), nn.Parameter(torch.full((d,), math.log(0.05)))
    if mode in bases:      # closed-form Hildreth-Houck moment init (rank R Burer-Monteiro lift)
        Pb = bases[mode](kd.psi).numpy().astype(np.float64)
        S_ = KR.hh_moments(Pb, kd.r.numpy().astype(np.float64)); G0, _ = KR.bm_lift(S_, Pb.shape[1], d, R, restarts=8, seed=sd)
        with torch.no_grad(): g.Gj.copy_(torch.as_tensor(G0, dtype=torch.float32))
        with torch.no_grad():
            xi0 = torch.cat([loglik_post(g(hs[j:j+8192], psi_pop[0][j:j+8192]), r_std[j:j+8192], Src() if Kc > 1 else src, logn)[1] for j in range(0, len(hs), 8192)]) if Kc == 1 else None
        if Kc == 1: print("   HH-init rank", R, "BLUP(nug .05) R2", score([torch.einsum("bdr,br->bd", g(hs, psi_pop[k]).double(), xi0).float().detach() for k in range(1, K)]))
    params = list(g.parameters()) + [logn] + (list(src.parameters()) if Kc > 1 else [])
    opt = torch.optim.Adam(params, lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); best, bs = 1e9, None
    for s in range(steps):
        i = torch.as_tensor(rng.choice(tr, 512)); l = -loglik_post(g(kd.h[i], kd.psi[i]), kd.r[i], src, logn)[0].mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sch.step()
        if (s + 1) % 500 == 0:
            with torch.no_grad(): vl = float(-torch.cat([loglik_post(g(kd.h[va[j:j+8192]], kd.psi[va[j:j+8192]]), kd.r[va[j:j+8192]], src, logn)[0] for j in range(0, len(va), 8192)]).mean())
            if vl < best: best, bs = vl, [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, b in zip(params, bs): p.copy_(b)
        xi = torch.cat([loglik_post(g(hs[j:j+8192], psi_pop[0][j:j+8192]), r_std[j:j+8192], src, logn)[1] for j in range(0, len(hs), 8192)])
        sc = score([torch.einsum("bdr,br->bd", g(hs, psi_pop[k]).double(), xi).float() for k in range(1, K)])
        lw, m, s_ = src.comps()
    print(name, mode, "K", Kc, "seed", sd, "val nll", round(best, 4), "R2", sc, "nug", (1e-3 + torch.exp(logn)).detach().numpy().round(3), f"{time.time()-t0:.0f}s", flush=True)
