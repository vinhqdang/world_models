"""Exact-structure oracle: cross-world kriging with the true conditional moments, estimated by Monte Carlo from the
parent's own draws of the disturbance (two folds over draws, so a draw never enters its own moments)."""
import numpy as np


def mc_kriging(truth, coords, tau_rel=1e-3):
    """truth (N, E, K, D) -> prediction (N, E, K, d) for coordinates ``coords`` (action 0 is the executed one)."""
    y = np.asarray(truth, np.float64)[..., coords]
    N, E, K, d = y.shape
    out = np.empty_like(y)
    out[:, :, 0] = y[:, :, 0]
    folds = [np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]
    for i in range(N):
        for f in (0, 1):
            test, ref = folds[f], folds[1 - f]
            yr = y[i, ref]                                     # (Er, K, d)
            mean = yr.mean(0)                                  # (K, d)
            c = yr - mean
            C00 = c[:, 0].T @ c[:, 0] / (len(yr) - 1)
            tau = tau_rel * np.trace(C00) / d + 1e-12
            Ainv = np.linalg.inv(C00 + tau * np.eye(d))
            for k in range(1, K):
                Ck0 = c[:, k].T @ c[:, 0] / (len(yr) - 1)
                out[i, test, k] = mean[k] + (y[i, test, 0] - mean[0]) @ (Ck0 @ Ainv).T
    return out


# ------------------------------------------------------------------------------------------------ learned-mean and nonlinear oracles
def _folds(E):
    return [np.arange(E) % 2 == 0, np.arange(E) % 2 == 1]


def mc_kriging_lm(truth, mu_pop, coords, tau_rel=1e-3):
    """oracle_kernel_lm*: the parent's own conditional covariance map (exact structure, estimated from the parent's draws in
    two folds) applied to the residual about the LEARNED mean: y_hat(a_k) = mu_hat(a_k) + L_k (y(a_0) - mu_hat(a_0))
    (experiments/cf_pilots/idt/diag4.py, 'learned-mean linear'). truth (N, E, K, D), mu_pop (N, K, D) -> (N, E, K, d)."""
    y = np.asarray(truth, np.float64)[..., coords]
    mu = np.asarray(mu_pop, np.float64)[..., coords]
    N, E, K, d = y.shape
    out = np.empty_like(y)
    out[:, :, 0] = y[:, :, 0]
    f = _folds(E)
    for i in range(N):
        for k_ in (0, 1):
            test, ref = f[k_], f[1 - k_]
            yr = y[i, ref]
            c = yr - yr.mean(0)
            C00 = c[:, 0].T @ c[:, 0] / (len(yr) - 1)
            Ainv = np.linalg.inv(C00 + (tau_rel * np.trace(C00) / d + 1e-12) * np.eye(d))
            for k in range(1, K):
                L = (c[:, k].T @ c[:, 0] / (len(yr) - 1)) @ Ainv
                out[i, test, k] = mu[i, k] + (y[i, test, 0] - mu[i, 0]) @ L.T
    return out


def _quad_features(u):
    d = u.shape[1]
    cols = [np.ones(len(u))] + [u[:, j] for j in range(d)] + [u[:, j] * u[:, k] for j in range(d) for k in range(j, d)]
    return np.stack(cols, 1)


def mc_quadratic(truth, coords, mu_pop=None, ridge_rel=1e-3):
    """oracle_quad*: per-parent QUADRATIC regression of y(a_k) on the centred executed outcome (two folds over draws), the
    nonlinear counterpart of mc_kriging. With ``mu_pop`` the centring and the target use the learned mean (oracle_quad*);
    without it the parent's exact conditional mean (oracle_quad_ex*; diag4.py 'exact-mean quadratic', all pairwise products)."""
    y = np.asarray(truth, np.float64)[..., coords]
    N, E, K, d = y.shape
    out = np.empty_like(y)
    out[:, :, 0] = y[:, :, 0]
    f = _folds(E)
    for i in range(N):
        for k_ in (0, 1):
            test, ref = f[k_], f[1 - k_]
            yr = y[i, ref]
            m0 = yr.mean(0) if mu_pop is None else np.asarray(mu_pop, np.float64)[i, :, :][..., coords]
            c0 = yr[:, 0] - m0[0]
            C00 = np.cov(yr[:, 0].T).reshape(d, d)
            F = _quad_features(c0)
            pen = np.diag(np.r_[0.0, np.ones(F.shape[1] - 1)]) * ridge_rel * len(F) * np.trace(C00) / d
            Ft = _quad_features(y[i, test, 0] - m0[0])
            for k in range(1, K):
                tgt = yr[:, k] - (0.0 if mu_pop is None else m0[k])
                B = np.linalg.solve(F.T @ F + pen + 1e-10 * np.eye(F.shape[1]), F.T @ tgt)
                out[i, test, k] = (0.0 if mu_pop is None else m0[k]) + Ft @ B
    return out


# ------------------------------------------------------------------------------------------------ structural Bayes
SQ2PI = np.sqrt(2 * np.pi)


def _logrho(law, e, mu, var):
    """log density of a driver value eps in (-1, 1) given the Gaussian driver N(mu, var) (laws of cwk.sim.law_eps;
    clipping atoms of gauss_clip are ignored). -inf outside (-1, 1)."""
    from scipy.special import ndtri
    sd = np.sqrt(var)
    inside = np.abs(e) < 1.0
    ec = np.clip(e, -1 + 1e-12, 1 - 1e-12)
    kind = law.get("kind", "tanh")
    if kind == "tanh":
        g = np.arctanh(ec)
        lj = -np.log1p(-ec ** 2)
    elif kind == "gated":
        g = ndtri((ec + 1) / 2)
        lj = -(np.log(2.0) - 0.5 * g ** 2 - np.log(SQ2PI))
    elif kind == "gauss_clip":
        c = law.get("c", 3.0)
        g = c * ec
        lj = np.log(c) * np.ones_like(ec)
    else:
        raise ValueError(kind)
    lp = -0.5 * ((g - mu) / sd) ** 2 - np.log(sd * SQ2PI) + lj
    return np.where(inside, lp, -np.inf)


GRID_PTS = {1: 101, 2: 31, 3: 13}


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def parent_posterior(J0, yres, s, laws, post):
    """Posterior mean of the driver vector for each draw of one parent under the linearised structural model
    y_res = J0 eps + N(0, s^2 I), drivers independent with the world's laws (gate prior included), exact over gate patterns:
    patterns with at least d active drivers are integrated over the (m - d) free coordinates on a midpoint grid with the d
    pivot coordinates solved exactly (noiseless limit), patterns with fewer than d drivers by their Laplace evidence.
    J0 (d, nd), yres (E, d) -> (E, nd)."""
    import itertools
    import scipy.linalg as sl
    d, nd = J0.shape
    E = len(yres)
    gated = [j for j in range(nd) if laws[j].get("kind") == "gated"]
    fixed = [j for j in range(nd) if laws[j].get("kind") != "gated"]
    logws, means = [], []
    for k in range(len(gated) + 1):
        for sub in itertools.combinations(gated, k):
            S = sorted(fixed + list(sub))
            m = len(S)
            lpri = sum(np.log(laws[j]["pi"]) if j in sub else np.log(1 - laws[j]["pi"]) for j in gated)
            em = np.zeros((E, nd))
            if m == 0:
                lev = -0.5 * (yres ** 2).sum(1) / s ** 2 - d * np.log(s * SQ2PI)
            elif m < d:
                A = J0[:, S]
                AtA = A.T @ A
                el = np.linalg.solve(AtA + 1e-12 * np.eye(m), A.T @ yres.T).T
                res = yres - el @ A.T
                lr = sum(_logrho(laws[j], el[:, a], post[j][0], post[j][1]) for a, j in enumerate(S))
                lev = (lr - 0.5 * np.linalg.slogdet(AtA + 1e-12 * np.eye(m))[1] - 0.5 * (res ** 2).sum(1) / s ** 2
                       - (d - m) * np.log(s * SQ2PI))
                em[:, S] = el
            else:
                A = J0[:, S]
                _, _, perm = sl.qr(A, pivoting=True)
                P, F = [S[i] for i in perm[:d]], [S[i] for i in perm[d:]]
                AP, AF = J0[:, P], (J0[:, F] if F else np.zeros((d, 0)))
                detP = abs(np.linalg.det(AP))
                nF = len(F)
                if nF:
                    ng = GRID_PTS[nF]
                    g1 = (np.arange(ng) + 0.5) / ng * 2 - 1
                    U = np.stack(np.meshgrid(*([g1] * nF), indexing="ij"), -1).reshape(-1, nF)
                    logvol = nF * np.log(2.0 / ng)
                else:
                    U, logvol = np.zeros((1, 0)), 0.0
                rhs = yres[:, None, :] - U @ AF.T[None] if nF else np.repeat(yres[:, None, :], 1, 1)
                eP = np.linalg.solve(AP[None, None], rhs[..., None])[..., 0]            # (E, G, d)
                lw = np.zeros(eP.shape[:2]) + logvol - np.log(detP + 1e-300)
                for a, j in enumerate(P):
                    lw = lw + _logrho(laws[j], eP[..., a], post[j][0], post[j][1])
                for a, j in enumerate(F):
                    lw = lw + _logrho(laws[j], U[None, :, a], post[j][0], post[j][1])
                lev = _logsumexp(lw, 1)
                W = np.exp(lw - np.where(np.isfinite(lev), lev, 0.0)[:, None])
                for a, j in enumerate(P):
                    em[:, j] = (W * np.where(np.isfinite(eP[..., a]), eP[..., a], 0.0)).sum(1)
                for a, j in enumerate(F):
                    em[:, j] = (W * U[None, :, a]).sum(1)
            logws.append(lpri + lev)
            means.append(em)
    lw = np.stack(logws, 1)
    mx = np.max(lw, 1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    W = np.exp(lw - mx)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-300)
    return np.einsum("ep,epj->ej", W, np.stack(means, 1))


def struct_bayes(pop, laws, coords, floor_rel=1e-3):
    """oracle_struct_bayes*: structural exact Bayes. Per parent the outcome is linearised in the drivers by finite
    differences of the simulator (pop['fd_y0'], pop['fd_J']); the posterior mean of the drivers given the executed outcome is
    computed by quadrature over the world's own driver laws (gates included) and transported through the linearised map of
    every query action. The misfit sd s of the linearisation (rms of the executed-branch residual over the parent's draws,
    privileged) is the likelihood noise. Returns (prediction (N, E, K, d), diagnostics)."""
    y = np.asarray(pop["truth"], np.float64)[..., coords]
    y0 = np.asarray(pop["fd_y0"], np.float64)[..., coords]
    Jm = np.asarray(pop["fd_J"], np.float64)[:, :, coords, :]
    eps_true = np.asarray(pop["eps"], np.float64)
    post = np.asarray(pop["posterior"], np.float64)
    N, E, K, d = y.shape
    pred = np.empty_like(y)
    pred[:, :, 0] = y[:, :, 0]
    lin = np.empty_like(y)
    lin[:, :, 0] = y[:, :, 0]
    s_all = np.empty(N)
    for i in range(N):
        ylin0 = y0[i, 0] + eps_true[i] @ Jm[i, 0].T
        s = float(np.sqrt(np.mean((y[i, :, 0] - ylin0) ** 2)))
        scale = float(y[i, :, 0].std(0).mean()) + 1e-12
        s = max(s, floor_rel * scale)
        s_all[i] = s
        eh = parent_posterior(Jm[i, 0], y[i, :, 0] - y0[i, 0], s, laws, post[i])
        for k in range(1, K):
            pred[i, :, k] = y0[i, k] + eh @ Jm[i, k].T
            lin[i, :, k] = y0[i, k] + eps_true[i] @ Jm[i, k].T
    return pred, dict(s_mean=float(s_all.mean()), lin=lin)
