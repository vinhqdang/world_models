"""EXPLORATORY: is the ICA failure optimisation or identification? NLL at the supervised G vs learned."""
import sys, pickle, glob, math
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import kernels as KR, metrics as M
torch.set_num_threads(2)
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
for p in sorted({"gain", prof}):
    F = np.c_[inn[:, :1], inn[:, 1:2] * m(psi[:, 1:], p)]; B = np.linalg.lstsq(F, r, rcond=None)[0]
    def G_of(ps, B=B, p=p):
        return np.stack([np.repeat(B[0][None], len(ps), 0), m(ps[:, 1:], p) @ B[1:]], -1)
    Gtr = G_of(psi); u = np.linalg.solve(Gtr, r[..., None])[..., 0]
    ok = np.abs(np.linalg.det(Gtr)) > 0.02
    print(p, "fit R2", (1 - ((r - F @ B) ** 2).sum() / (r ** 2).sum()).round(3), "kurt u (well-cond):", ((u[ok]**4).mean(0)/(u[ok]**2).mean(0)**2-3).round(2), "corr(u,inn)", [np.corrcoef(u[ok][:, j], inn[ok][:, j])[0,1].round(3) for j in range(2)])
    Gs = G_of(psi_pop[0].numpy().astype(np.float64))
    for tau in (1e-3, 1e-2, 3e-2, 1e-1):
        A = np.einsum("bdr,bds->brs", Gs, Gs) + tau * np.eye(2)
        w = np.linalg.solve(A, np.einsum("bdr,bd->br", Gs, r_std.numpy().astype(np.float64))[..., None])[..., 0]
        print("   supervised(innov) G, profile", p, "tau", tau, "R2", score([np.einsum("bdr,br->bd", G_of(psi_pop[k].numpy().astype(np.float64)), w) for k in range(1, K)]))
    # noise level: residual not explained
    print("   unexplained sd per coord", (r - F @ B).std(0).round(3))
