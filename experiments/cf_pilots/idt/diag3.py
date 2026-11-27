"""EXPLORATORY: Bayes (non-Gaussian prior) vs linear abduction with a supervised (privileged) G: is there nonlinear headroom?"""
import sys, pickle, glob, math
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import kernels as KR, metrics as M
name = sys.argv[1]; prof = name.split("_")[2]
Z = pickle.load(open(f"{name}_s0.pkl", "rb"))
kd, hs, psi_pop, r_std, mu_pop, truth, coords = Z["kd"], Z["hs"], Z["psi_pop"], Z["r_std"], Z["mu_pop"], Z["truth"], Z["coords"]
col = np.load(glob.glob(f"/home/user/world_models/results/cwk/data/{name}_col_*.npz")[0])
eps = col["eps"].reshape(-1, 2)
N, E, K, D = truth.shape; d = len(coords)
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
base = np.repeat(mu_pop[:, None], E, 1)
def score(deltas):
    pred = base.copy()
    for k in range(1, K):
        pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + kd.sigma * np.asarray(deltas[k - 1]).reshape(N, E, d)
    return round(100 * (1 - ev.loss(pred).mean() / ev.var), 2)
r = kd.r.numpy().astype(np.float64); psi = kd.psi.numpy().astype(np.float64)
h = kd.h.numpy().astype(np.float64); Hb = np.c_[h, np.ones(len(h))]
inn = eps - Hb @ np.linalg.lstsq(Hb, eps, rcond=None)[0]; inn = inn / inn.std(0)
def m(xx, p):
    if p == "sat": return np.tanh(3 * xx) / np.tanh(3)
    if p == "deadzone": return np.sign(xx) * np.maximum(np.abs(xx) - .4, 0) / .6
    return xx
F = np.c_[inn[:, :1], inn[:, 1:2] * m(psi[:, 1:], prof)]; B = np.linalg.lstsq(F, r, rcond=None)[0]
s2 = ((r - F @ B) ** 2).mean()
def G_of(ps): return np.stack([np.repeat(B[0][None], len(ps), 0), m(ps[:, 1:], prof) @ B[1:]], -1)
ps = [p.numpy().astype(np.float64) for p in psi_pop]; Gs = G_of(ps[0]); rs = r_std.numpy().astype(np.float64)
for tau in (1e-2, s2, 3e-2):
    A = np.einsum("bdr,bds->brs", Gs, Gs) + tau * np.eye(2)
    w = np.linalg.solve(A, np.einsum("bdr,bd->br", Gs, rs)[..., None])[..., 0]
    print("linear tau", round(tau, 4), score([np.einsum("bdr,br->bd", G_of(ps[k]), w) for k in range(1, K)]))
rng = np.random.default_rng(0); part = inn[rng.choice(len(inn), 3000, replace=False)]          # prior particles (M, 2)
for s2f in (0.5, 1.0, 2.0):
    w = np.empty((len(rs), 2))
    for i in range(0, len(rs), 2048):
        g = Gs[i:i + 2048]; pred = np.einsum("bdr,mr->bmd", g, part)
        ll = -0.5 * ((rs[i:i + 2048, None] - pred) ** 2).sum(-1) / (s2 * s2f)
        ll -= ll.max(1, keepdims=True); p = np.exp(ll); p /= p.sum(1, keepdims=True)
        w[i:i + 2048] = p @ part
    print("Bayes particle posterior mean, noise var x", s2f, score([np.einsum("bdr,br->bd", G_of(ps[k]), w) for k in range(1, K)]), flush=True)
