"""EXPLORATORY synthetic: overcomplete sparse (intermittent) actuator faults, d = 2 outputs, r = 3 drivers.
r(a) = J [ xi0 (1,-1) + xi1 (x1, 0) + xi2 (0, x2) ] + noise; xi_j = b_j u_j, b ~ Bern(pi), u ~ U(-1,1) (independent).
Arms: oracle linear BLUP (true K), HH moment + BLUP, learned independent-factor (MoG) Bayes transport, oracle Bayes (true G, true prior via particles)."""
import warnings; warnings.filterwarnings('ignore')
import sys, math, itertools, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/user/world_models")
from cwk import kernels as KR
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
pi, noise, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); src = sys.argv[4] if len(sys.argv) > 4 else "bern"; Rr = int(sys.argv[5]) if len(sys.argv) > 5 else 3; rng = np.random.default_rng(seed)
d, R, n, nt = 2, Rr, 80000, 20000
J = np.array([[1.0, 0.3], [-0.2, 0.8]])
def Gtrue(x):                                  # (B, d, R)
    P = np.zeros((len(x), 2, 3)); P[:, 0, 0] = 1; P[:, 1, 0] = -1; P[:, 0, 1] = x[:, 0]; P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)[:, :, :R] if R == 3 else np.einsum("de,ber->bdr", J, np.stack([P[:, :, 0], P[:, :, 1] - P[:, :, 2]], -1))
def draw(m):
    if src == "tanh":
        e = np.tanh(2.0 * rng.standard_normal((m, R))); return e / 0.78
    return (rng.random((m, R)) < pi) * rng.uniform(-1, 1, (m, R)) / math.sqrt(pi / 3)       # unit variance
x = rng.uniform(-1, 1, (n, 2)); xi = draw(n); r = np.einsum("bdr,br->bd", Gtrue(x), xi) + noise * rng.standard_normal((n, d))
xt, xt2, xit = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)), draw(nt)
rt = np.einsum("bdr,br->bd", Gtrue(xt), xit) + noise * rng.standard_normal((nt, d)); tgt = np.einsum("bdr,br->bd", Gtrue(xt2), xit)
def R2(p): return round(100 * (1 - ((p - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()), 2)
def blup(Gs, Gt, nug):
    A = Gs @ Gs.transpose(0, 2, 1) + nug * np.eye(d)
    return np.einsum("bdr,br->bd", Gt, np.einsum("bdr,bd->br", Gs, np.linalg.solve(A, rt[..., None])[..., 0]))
print(f"src={src} r={R} pi={pi} noise={noise} seed={seed}: oracle linear BLUP", R2(blup(Gtrue(xt), Gtrue(xt2), noise ** 2)), end=" | ", flush=True)
psi = lambda z: np.c_[np.ones(len(z)), z]
S = KR.hh_moments(psi(x), r); G0, _ = KR.bm_lift(S, 3, d, R, restarts=8, seed=seed)
Gh = lambda z: np.einsum("bq,qdr->bdr", psi(z), G0)
print("HH+BLUP", R2(blup(Gh(xt), Gh(xt2), max(noise ** 2, 1e-3))), end=" | ")
part = draw(4000)                               # oracle Bayes by particles
def bayes(Gs, Gt):
    out = np.empty((nt, d))
    for i in range(0, nt, 1000):
        pred = np.einsum("bdr,mr->bmd", Gs[i:i+1000], part); ll = -0.5 * ((rt[i:i+1000, None] - pred) ** 2).sum(-1) / max(noise, 0.02) ** 2
        w = np.exp(ll - ll.max(1, keepdims=True)); w /= w.sum(1, keepdims=True); out[i:i+1000] = np.einsum("bdr,br->bd", Gt[i:i+1000], w @ part)
    return out
print("oracle Bayes (true G, prior)", R2(bayes(Gtrue(xt), Gtrue(xt2))), end=" | ", flush=True)
# learned independent-factor model: G = sum_q psi_q G_q (init HH), sources MoG K=3 (zero-mean comps: scale mixture) + diag noise
Kc = 4; combos = torch.tensor(list(itertools.product(range(Kc), repeat=R)))
Gq = nn.Parameter(torch.tensor(G0)); ls = nn.Parameter(torch.full((R, Kc), -1.0)); mm = nn.Parameter(torch.linspace(-1.2, 1.2, Kc).repeat(R, 1)); lw = nn.Parameter(torch.zeros(R, Kc)); logn = nn.Parameter(torch.tensor(math.log(0.02)))
def model(P, rr):
    G = torch.einsum("bq,qdr->bdr", P, Gq); ar = torch.arange(R); lwc = torch.log_softmax(lw, -1)[ar, combos].sum(-1); s = torch.exp(2 * ls)[ar, combos]
    Gs = G[:, None] * s[None, :, None, :]; cov = Gs @ G[:, None].transpose(-1, -2) + (1e-4 + torch.exp(logn)) * torch.eye(d)
    mc = mm[ar, combos]; dev = rr[:, None] - torch.einsum("bdr,cr->bcd", G, mc)
    L = torch.linalg.cholesky(cov); sol = torch.cholesky_solve(dev[..., None], L)[..., 0]
    lp = lwc[None] - 0.5 * (dev * sol).sum(-1) - torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return torch.logsumexp(lp, 1), (torch.softmax(lp, 1)[..., None] * (mc[None] + torch.einsum("bcdr,bcd->bcr", Gs, sol))).sum(1)
P, rr = torch.tensor(psi(x)), torch.tensor(r); opt = torch.optim.Adam([Gq, ls, lw, logn, mm], lr=3e-3)
for s_ in range(2500):
    i = torch.as_tensor(rng.choice(n, 512)); l = -model(P[i], rr[i])[0].mean(); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    xh = torch.cat([model(torch.tensor(psi(xt))[i:i+4000], torch.tensor(rt)[i:i+4000])[1] for i in range(0, nt, 4000)])
    print("learned independent-factor Bayes", R2(torch.einsum("bdr,br->bd", torch.einsum("bq,qdr->bdr", torch.tensor(psi(xt2)), Gq), xh).numpy()))
