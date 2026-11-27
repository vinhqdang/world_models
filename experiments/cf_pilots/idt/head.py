"""EXPLORATORY pilot: independent-driver (non-Gaussian) head vs HH / flow on cached residuals."""
import sys, pickle, math, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch, torch.nn as nn
from cwk import kernels as KR, metrics as M
from cwk.predictor import mlp
torch.set_num_threads(2)
name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
Z = pickle.load(open(f"{name}_s{seed}.pkl", "rb"))
kd, hs, psi_pop, r_std, mu_pop, truth, coords = Z["kd"], Z["hs"], Z["psi_pop"], Z["r_std"], Z["mu_pop"], Z["truth"], Z["coords"]
N, E, K, D = truth.shape; d = len(coords)
idx = Z["groups"][Z["primary"]]
ev = M.Evaluator(truth, Z["weights"], idx, n_seeds=1)
base = np.repeat(mu_pop[:, None], E, 1)
def score(deltas):
    pred = base.copy()
    for k in range(1, K):
        pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + kd.sigma * deltas[k - 1].numpy().reshape(N, E, d)
    return 100 * (1 - ev.loss(pred).mean() / ev.var)
print("oracle_kernel", 100 * (1 - Z["det"][Z["primary"]]["oracle_kernel"].mean() / ev.var), "rr_exact", 100 * (1 - Z["det"][Z["primary"]]["rr_exact"].mean() / ev.var))
rr = base + (truth[:, :, :1] - base[:, :, :1]); print("rr_learned", 100 * (1 - ev.loss(rr).mean() / ev.var), flush=True)

# ---- HH global with given psi
def hh(psi_tr, psi_p, rank=4):
    S = KR.hh_moments(psi_tr.numpy().astype(np.float64), kd.r.numpy().astype(np.float64))
    q = psi_tr.shape[1]
    G, _ = KR.bm_lift(S, q, d, rank, restarts=6, seed=seed)
    G = torch.as_tensor(G)
    Gs = torch.einsum("bq,qdr->bdr", psi_p[0].double(), G)
    A = Gs.transpose(1, 2) @ Gs + KR.TAU * torch.eye(rank, dtype=torch.float64)
    w = torch.linalg.solve(A, torch.einsum("bdr,bd->br", Gs, r_std.double())[..., None])[..., 0]
    return [torch.einsum("bdr,br->bd", torch.einsum("bq,qdr->bdr", psi_p[k].double(), G), w).float() for k in range(1, K)]
print("hh_global(lin)", score(hh(kd.psi, psi_pop)), flush=True)
def cub(p): return torch.cat([p, p[:, 1:] ** 3], -1)
print("hh_global(1,x,x^3)", score(hh(cub(kd.psi), [cub(p) for p in psi_pop], rank=6)), flush=True)
def quad(p): return torch.cat([p, p[:, 1:] ** 2, p[:, 1:] ** 3], -1)
print("hh_global(1,x,x^2,x^3)", score(hh(quad(kd.psi), [quad(p) for p in psi_pop], rank=6)), flush=True)

# ---- ICA head
class MoG(nn.Module):
    """per-source 1-D mixture of Gaussians, learnable (location, scale, weights)."""
    def __init__(self, r, k=6):
        super().__init__()
        self.m = nn.Parameter(torch.linspace(-1.5, 1.5, k).repeat(r, 1)); self.ls = nn.Parameter(torch.full((r, k), -0.7)); self.lw = nn.Parameter(torch.zeros(r, k))
    def logp(self, u):
        z = (u[..., None] - self.m) / torch.exp(self.ls)
        return torch.logsumexp(torch.log_softmax(self.lw, -1) - 0.5 * z * z - self.ls - 0.5 * math.log(2 * math.pi), -1)

class Gnet(nn.Module):
    def __init__(self, dh, q, d, width=256, depth=3, use_h=True):
        super().__init__()
        self.d, self.use_h = d, use_h
        self.net = mlp((dh if use_h else 0) + q, d * d, width, depth)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.3); self.net[-1].bias.copy_(torch.eye(d).flatten() * 0.7 + 0.1 * torch.randn(d * d))
    def forward(self, h, psi):
        x = torch.cat([h, psi], -1) if self.use_h else psi
        return self.net(x).view(-1, self.d, self.d)

def fit_ica(gauss=False, use_h=True, steps=steps, lr=2e-3, batch=512, sd=seed):
    torch.manual_seed(sd); rng = np.random.default_rng(sd)
    eps_ids = np.unique(kd.ep); val = set(rng.choice(eps_ids, len(eps_ids) // 10, replace=False).tolist())
    isv = np.array([e in val for e in kd.ep]); tr, va = np.where(~isv)[0], np.where(isv)[0]
    net, base_d = Gnet(kd.dh, kd.q, d, use_h=use_h), MoG(d)
    params = list(net.parameters()) + ([] if gauss else list(base_d.parameters()))
    opt = torch.optim.Adam(params, lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    def nll(i):
        G = net(kd.h[i], kd.psi[i]).double(); G = G + 1e-3 * torch.eye(d, dtype=torch.float64) * torch.sign(torch.linalg.det(G))[:, None, None]
        u = torch.linalg.solve(G, kd.r[i].double()[..., None])[..., 0]
        lp = (-0.5 * u * u - 0.5 * math.log(2 * math.pi)) if gauss else base_d.logp(u.float()).double()
        return -(lp.sum(-1)) + torch.log(torch.linalg.det(G).abs() + 1e-9)
    best, bs = math.inf, None
    for s in range(steps):
        i = torch.as_tensor(rng.choice(tr, batch)); loss = nll(i).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step()
        if (s + 1) % 250 == 0:
            with torch.no_grad(): vl = float(nll(torch.as_tensor(va)).mean())
            if vl < best: best, bs = vl, [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, b in zip(params, bs): p.copy_(b)
    return net, base_d, best

@torch.no_grad()
def transport(net, tau=KR.TAU):
    Gs = net(hs, psi_pop[0]).double()
    A = Gs.transpose(1, 2) @ Gs + tau * torch.eye(d, dtype=torch.float64)
    w = torch.linalg.solve(A, torch.einsum("bdr,bd->br", Gs, r_std.double())[..., None])[..., 0]
    return [torch.einsum("bdr,br->bd", net(hs, psi_pop[k]).double(), w).float() for k in range(1, K)], w

for label, kw in [("ica(h,a)", dict()), ("ica(a only)", dict(use_h=False)), ("gauss-free(h,a)", dict(gauss=True))]:
    for sd in (seed, seed + 100):
        t0 = time.time(); net, bd, vl = fit_ica(sd=sd, **kw); dl, w = transport(net)
        ku = ((w ** 4).mean(0) / (w ** 2).mean(0) ** 2 - 3).numpy().round(2)
        for tau in (1e-3, 1e-2, 3e-2):
            print(label, "seed", sd, "tau", tau, "R2", round(score(transport(net, tau)[0]), 2), "val nll", round(vl, 4), "kurt(u)", ku, f"{time.time() - t0:.0f}s", flush=True)
