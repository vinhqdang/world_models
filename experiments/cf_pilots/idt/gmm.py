# Baseline: HH lift (gauge-arbitrary rotation) + dependent full-covariance Gaussian-mixture prior on xi (no independence, no ICA), posterior-mean transport
import sys, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/user/world_models")
from cwk import kernels as KR
import warnings; warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
pi, noise, seed, Kc, src = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(seed); torch.manual_seed(seed)
d, R, n, nt = 2, 3, 80000, 20000
J = np.array([[1.0, 0.3], [-0.2, 0.8]])
def Gtrue(x):
    P = np.zeros((len(x), 2, 3)); P[:, 0, 0] = 1; P[:, 1, 0] = -1; P[:, 0, 1] = x[:, 0]; P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)
def draw(m):
    if src == "tanh": return np.tanh(2.0 * rng.standard_normal((m, R))) / 0.78
    return (rng.random((m, R)) < pi) * rng.uniform(-1, 1, (m, R)) / math.sqrt(pi / 3)
x = rng.uniform(-1, 1, (n, 2)); xi = draw(n); r = np.einsum("bdr,br->bd", Gtrue(x), xi) + noise * rng.standard_normal((n, d))
xt, xt2, xit = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)), draw(nt)
rt = np.einsum("bdr,br->bd", Gtrue(xt), xit) + noise * rng.standard_normal((nt, d)); tgt = np.einsum("bdr,br->bd", Gtrue(xt2), xit)
def R2(p): return round(100 * (1 - ((p - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()), 2)
psi = lambda z: np.c_[np.ones(len(z)), z]
S_ = KR.hh_moments(psi(x), r); G0, _ = KR.bm_lift(S_, 3, d, R, restarts=8, seed=seed)
Gq = torch.tensor(G0)
m = nn.Parameter(torch.randn(Kc, R)); Lr = nn.Parameter(torch.zeros(Kc, R, R)); ld = nn.Parameter(torch.full((Kc, R), -1.0)); lw = nn.Parameter(torch.zeros(Kc)); logn = nn.Parameter(torch.tensor(math.log(0.02)))
def model(P, rr, learnG=None):
    G = torch.einsum("bq,qdr->bdr", P, Gq if learnG is None else learnG)
    L = torch.tril(Lr, -1) + torch.diag_embed(torch.exp(ld)); S = L @ L.transpose(-1, -2)
    GS = torch.einsum("bdr,crs->bcds", G, S); cov = GS @ G[:, None].transpose(-1, -2) + (1e-4 + torch.exp(logn)) * torch.eye(d)
    dev = rr[:, None] - torch.einsum("bdr,cr->bcd", G, m); Lc = torch.linalg.cholesky(cov); sol = torch.cholesky_solve(dev[..., None], Lc)[..., 0]
    lp = torch.log_softmax(lw, 0)[None] - 0.5 * (dev * sol).sum(-1) - torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    return torch.logsumexp(lp, 1), (torch.softmax(lp, 1)[..., None] * (m[None] + torch.einsum("bcds,bcd->bcs", GS, sol))).sum(1)
P, rr = torch.tensor(psi(x)), torch.tensor(r); opt = torch.optim.Adam([m, Lr, ld, lw, logn], lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
for s_ in range(3000):
    i = torch.as_tensor(rng.choice(n, 1024)); l = -model(P[i], rr[i])[0].mean(); opt.zero_grad(); l.backward(); opt.step(); sched.step()
with torch.no_grad():
    xh = torch.cat([model(torch.tensor(psi(xt))[i:i+4000], torch.tensor(rt)[i:i+4000])[1] for i in range(0, nt, 4000)])
    print(f"src={src} pi={pi} seed={seed} K={Kc}: dependent-GMM prior on HH lift (no ICA) R2", R2(torch.einsum("bdr,br->bd", torch.einsum("bq,qdr->bdr", torch.tensor(psi(xt2)), Gq), xh).numpy()), "train nll", round(float(l), 3))
