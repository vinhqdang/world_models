"""EXPLORATORY: basis-structured ICA likelihood (global G = sum_j phi_j(a) G_j, square) vs HH with the same basis."""
import sys, pickle, math, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch, torch.nn as nn
from cwk import kernels as KR, metrics as M
torch.set_num_threads(2)
name = sys.argv[1]
Z = pickle.load(open(f"{name}_s0.pkl", "rb"))
kd, hs, psi_pop, r_std, mu_pop, truth, coords = Z["kd"], Z["hs"], Z["psi_pop"], Z["r_std"], Z["mu_pop"], Z["truth"], Z["coords"]
N, E, K, D = truth.shape; d = len(coords)
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
base = np.repeat(mu_pop[:, None], E, 1)
def score(deltas):
    pred = base.copy()
    for k in range(1, K):
        pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + kd.sigma * deltas[k - 1].numpy().reshape(N, E, d)
    return round(100 * (1 - ev.loss(pred).mean() / ev.var), 2)
bases = {"lin": lambda p: p, "cub": lambda p: torch.cat([p, p[:, 1:] ** 3], -1), "quadcub": lambda p: torch.cat([p, p[:, 1:] ** 2, p[:, 1:] ** 3], -1),
         "poly5": lambda p: torch.cat([p] + [p[:, 1:] ** k for k in (2, 3, 4, 5)], -1)}
class MoG(nn.Module):
    def __init__(self, r, k=6):
        super().__init__()
        self.m = nn.Parameter(torch.linspace(-1.5, 1.5, k).repeat(r, 1)); self.ls = nn.Parameter(torch.full((r, k), -0.7)); self.lw = nn.Parameter(torch.zeros(r, k))
    def logp(self, u):
        z = (u[..., None] - self.m) / torch.exp(self.ls)
        return torch.logsumexp(torch.log_softmax(self.lw, -1) - 0.5 * z * z - self.ls - 0.5 * math.log(2 * math.pi), -1)
rng0 = np.random.default_rng(0); eps_ids = np.unique(kd.ep); val = set(rng0.choice(eps_ids, len(eps_ids) // 10, replace=False).tolist())
isv = np.array([e in val for e in kd.ep]); tr, va = np.where(~isv)[0], np.where(isv)[0]
def fit(bname, gauss, sd, steps=3000, noise=True):
    torch.manual_seed(sd); rng = np.random.default_rng(sd); f = bases[bname]; P = f(kd.psi).double(); q = P.shape[1]
    Gj = nn.Parameter(0.3 * torch.randn(q, d, d, dtype=torch.float64)); bd = MoG(d).double()
    params = [Gj] + ([] if gauss else list(bd.parameters())); opt = torch.optim.Adam(params, lr=5e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    def nll(i):
        G = torch.einsum("bq,qdr->bdr", P[i], Gj); det = torch.linalg.det(G)
        G = G + 1e-3 * torch.eye(d, dtype=torch.float64) * torch.where(det >= 0, 1.0, -1.0)[:, None, None]
        u = torch.linalg.solve(G, kd.r[i].double()[..., None])[..., 0]
        lp = (-0.5 * u * u - 0.5 * math.log(2 * math.pi)) if gauss else bd.logp(u)
        return -lp.sum(-1) + torch.log(torch.linalg.det(G).abs() + 1e-9)
    for s in range(steps):
        i = torch.as_tensor(rng.choice(tr, 1024)); l = nll(i).mean(); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sch.step()
    with torch.no_grad():
        vl = float(nll(torch.as_tensor(va)).mean()); out = {}
        for tau in (1e-3, 3e-2):
            Gs = torch.einsum("bq,qdr->bdr", f(psi_pop[0]).double(), Gj)
            A = Gs.transpose(1, 2) @ Gs + tau * torch.eye(d, dtype=torch.float64)
            w = torch.linalg.solve(A, torch.einsum("bdr,bd->br", Gs, r_std.double())[..., None])[..., 0]
            out[tau] = score([torch.einsum("bdr,br->bd", torch.einsum("bq,qdr->bdr", f(psi_pop[k]).double(), Gj), w).float() for k in range(1, K)])
    return vl, out
for bname in sys.argv[2:]:
    for gauss in (False, True):
        for sd in (0, 1, 2):
            t0 = time.time(); vl, out = fit(bname, gauss, sd)
            print(name, bname, "gauss" if gauss else "ICA", "seed", sd, "val nll", round(vl, 4), "R2", out, f"{time.time()-t0:.0f}s", flush=True)
