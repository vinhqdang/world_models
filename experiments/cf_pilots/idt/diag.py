"""EXPLORATORY: supervised ceilings using the saved drivers eps (privileged) to locate the headroom."""
import sys, pickle, glob
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import kernels as KR, metrics as M
from cwk.predictor import mlp
torch.set_num_threads(2)
name = sys.argv[1]
Z = pickle.load(open(f"{name}_s0.pkl", "rb"))
kd, hs, psi_pop, r_std, mu_pop, truth, coords = Z["kd"], Z["hs"], Z["psi_pop"], Z["r_std"], Z["mu_pop"], Z["truth"], Z["coords"]
col = np.load(glob.glob(f"/home/user/world_models/results/cwk/data/{name}_col_*.npz")[0])
eps = col["eps"].reshape(-1, 2)                       # (n*T, 2) aligned with flatten_transitions
N, E, K, D = truth.shape; d = len(coords)
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
base = np.repeat(mu_pop[:, None], E, 1)
def score(deltas):
    pred = base.copy()
    for k in range(1, K):
        pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + kd.sigma * np.asarray(deltas[k - 1]).reshape(N, E, d)
    return round(100 * (1 - ev.loss(pred).mean() / ev.var), 2)
r = kd.r.numpy().astype(np.float64); psi = kd.psi.numpy().astype(np.float64); x = psi[:, 1:]
print("axes", Z["axes"], "q", psi.shape, "kurt eps", ((eps**4).mean(0)/(eps**2).mean(0)**2-3).round(2), "corr eps", np.corrcoef(eps.T)[0,1].round(3))
# innovation of eps given H: regress eps on h (linear) to see how much is absorbed
h = kd.h.numpy().astype(np.float64); Hb = np.c_[h, np.ones(len(h))]
be = np.linalg.lstsq(Hb, eps, rcond=None)[0]; inn = eps - Hb @ be
print("frac var eps explained by linear H:", (1 - inn.var(0) / eps.var(0)).round(3), "kurt innov", ((inn**4).mean(0)/(inn**2).mean(0)**2-3).round(2))
def feats(e, ps, prof=None):
    xx = ps[:, 1:]
    if prof == "sat": xx = np.tanh(3 * xx) / np.tanh(3)
    if prof == "deadzone": xx = np.sign(xx) * np.maximum(np.abs(xx) - .4, 0) / .6
    return np.c_[e[:, :1], e[:, 1:2] * xx]       # true structure: offset column, gain driver times m(x_axis)
def trans(G_fn, tag):
    # r ~ G(a) xi; transport with ridge pseudo-inverse of G(a0)
    Gs = G_fn(0); out = []
    A = np.einsum("bdr,bds->brs", Gs, Gs) + KR.TAU * np.eye(Gs.shape[-1])
    w = np.linalg.solve(A, np.einsum("bdr,bd->br", Gs, r_std.numpy().astype(np.float64))[..., None])[..., 0]
    for k in range(1, K): out.append(np.einsum("bdr,br->bd", G_fn(k), w))
    print(tag, score(out), flush=True)
for prof in ([None] + ([name.split("_")[2]] if name.split("_")[2] in ("sat", "deadzone") else [])):
    F = feats(eps, psi, prof); B = np.linalg.lstsq(F, r, rcond=None)[0]          # (1 + p, d)
    print(prof, "supervised global fit R2 of residual:", (1 - ((r - F @ B) ** 2).sum() / (r ** 2).sum()).round(3))
    def G_fn(k, prof=prof, B=B):
        ps = psi_pop[k].numpy().astype(np.float64); xx = ps[:, 1:]
        if prof == "sat": xx = np.tanh(3 * xx) / np.tanh(3)
        if prof == "deadzone": xx = np.sign(xx) * np.maximum(np.abs(xx) - .4, 0) / .6
        g0 = np.repeat(B[0][None], len(ps), 0); g1 = xx @ B[1:]
        return np.stack([g0, g1], -1)
    trans(G_fn, f"supervised-G global, profile={prof}:")
# innovation version: use innovations as regressors (what the residual can see)
F = feats(inn, psi, None); B = np.linalg.lstsq(F, r, rcond=None)[0]
print("innov fit R2", (1 - ((r - F @ B) ** 2).sum() / (r ** 2).sum()).round(3))
