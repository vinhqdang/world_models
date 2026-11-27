"""EXPLORATORY: mean-error-aware ('epistemic') cross-world kriging with an ensemble of M mean predictors.
G: global HH (linear psi, rank 2 BM lift) from seed-0 cross-fitted residuals; nugget grid."""
import sys, pickle, glob
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import kernels as KR, metrics as M
name = sys.argv[1]
Zs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{name}_s*.pkl"))]
Z = Zs[0]; kd, psi_pop, truth, coords = Z["kd"], Z["psi_pop"], Z["truth"], Z["coords"]
N, E, K, D = truth.shape; d = len(coords); Mm = len(Zs)
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
y = truth[..., coords]; sig = kd.sigma
mus = np.stack([z["mu_pop"][:, :, coords] for z in Zs])                 # (M, N, K, d)
def sc(pred):
    full = np.array(truth); full[..., coords] = pred; return round(100 * (1 - ev.loss(full).mean() / ev.var), 2)
S = KR.hh_moments(kd.psi.numpy().astype(np.float64), kd.r.numpy().astype(np.float64)); G0, _ = KR.bm_lift(S, 3, d, 2, restarts=8, seed=0)
ps = [p.numpy().astype(np.float64).reshape(N, E, -1)[:, 0] for p in psi_pop]            # (N, q) per k
G = [np.einsum("nq,qdr->ndr", p, G0) for p in ps]                                      # (N, d, R) per k (standardised units)
def run(mu, Ce=None, nug=0.03, gam=1.0):
    out = np.empty_like(y); out[:, :, 0] = y[:, :, 0]
    r0 = (y[:, :, 0] - mu[:, None, 0]) / sig
    for k in range(1, K):
        K00 = G[0] @ G[0].transpose(0, 2, 1) + nug * np.eye(d); Kk0 = G[k] @ G[0].transpose(0, 2, 1)
        if Ce is not None:
            K00 = K00 + gam * Ce[:, 0, 0]; Kk0 = Kk0 + gam * Ce[:, k, 0]
        L = Kk0 @ np.linalg.inv(K00)
        out[:, :, k] = mu[:, None, k] + sig * np.einsum("nde,nme->nmd", L, r0)
    return sc(out)
print(name, "members", Mm)
for nug in (1e-3, 0.03, 0.1): print("single mean (seed 0), nugget", nug, run(mus[0], nug=nug))
mbar = mus.mean(0)
for nug in (1e-3, 0.03, 0.1): print("ensemble mean, nugget", nug, run(mbar, nug=nug))
dev = (mus - mbar) / sig                                                               # (M, N, K, d)
Ce = np.einsum("mnkd,mnle->nkdle", dev, dev) / (Mm - 1) / Mm                           # cov of the ENSEMBLE MEAN error, per parent: (N,K,d,K,d)
Ce = Ce.transpose(0, 1, 3, 2, 4)                                                       # (N, K, K, d, d)
Cs = np.einsum("mnkd,mnle->nkdle", dev, dev).transpose(0, 1, 3, 2, 4) / (Mm - 1)       # single-member error cov
for gam in (1, 3, 10, 30):
    print("ensemble mean + epistemic kernel, gamma", gam, [run(mbar, Ce, nug=n_, gam=gam) for n_ in (1e-3, 0.03)])
for gam in (1, 3, 10): print("single mean + epistemic kernel (member cov), gamma", gam, run(mus[0], Cs, nug=0.03, gam=gam))
e = mbar - y.mean(1); e0 = mus[0] - y.mean(1)
print("rms mean error / within sd: single", np.sqrt((e0**2).mean() / y.var(1).mean()).round(3), "ensemble", np.sqrt((e**2).mean() / y.var(1).mean()).round(3), "ensemble spread rms / within sd", np.sqrt(((mus-mbar)**2).mean() / y.var(1).mean()).round(3))
