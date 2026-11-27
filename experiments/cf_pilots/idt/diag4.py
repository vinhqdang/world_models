"""EXPLORATORY: oracle per-parent linear map with LEARNED means; and per-parent nonlinear (quadratic) oracle."""
import sys, pickle
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import metrics as M
name = sys.argv[1]
Z = pickle.load(open(f"{name}_s0.pkl", "rb"))
mu_pop, truth, coords = Z["mu_pop"], Z["truth"], Z["coords"]
N, E, K, D = truth.shape; d = len(coords)
ev = M.Evaluator(truth, Z["weights"], Z["groups"][Z["primary"]], n_seeds=1)
y = truth[..., coords]
def sc(pred):
    full = np.array(truth); full[..., coords] = pred; return round(100 * (1 - ev.loss(full).mean() / ev.var), 2)
folds = [np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]
for mode in ("exact-mean linear", "learned-mean linear", "exact-mean quadratic", "exact-mean linear, tau=3e-2"):
    out = np.empty_like(y); out[:, :, 0] = y[:, :, 0]
    for i in range(N):
        for f in (0, 1):
            te, rf = folds[f], folds[1 - f]; yr = y[i, rf]; mean = yr.mean(0); c = yr - mean
            C00 = c[:, 0].T @ c[:, 0] / (len(yr) - 1); tau = (3e-2 if "tau" in mode else 1e-3) * np.trace(C00) / d
            Ai = np.linalg.inv(C00 + tau * np.eye(d))
            for k in range(1, K):
                L = (c[:, k].T @ c[:, 0] / (len(yr) - 1)) @ Ai
                if mode.startswith("learned"):
                    out[i, te, k] = mu_pop[i, k, coords] + (y[i, te, 0] - mu_pop[i, 0, coords]) @ L.T
                elif "quadratic" in mode:
                    def feat(u): return np.c_[np.ones(len(u)), u, u[:, :1] * u, (u[:, 1:] ** 2)][:, :1 + d + d * (d + 1) // 2]
                    Fm = feat(c[:, 0]); Bq = np.linalg.solve(Fm.T @ Fm + 1e-3 * len(Fm) * np.diag(np.r_[0, np.ones(Fm.shape[1] - 1)]) * np.trace(C00) / d, Fm.T @ yr[:, k])
                    out[i, te, k] = feat(y[i, te, 0] - mean[0]) @ Bq
                else:
                    out[i, te, k] = mean[k] + (y[i, te, 0] - mean[0]) @ L.T
    print(name, mode, sc(out), flush=True)
# mean error size relative to outcome sd
e = mu_pop[:, :, coords] - y.mean(1); print("rms mean error / within-parent sd:", np.sqrt((e ** 2).mean()) / np.sqrt(y.var(1).mean()))
