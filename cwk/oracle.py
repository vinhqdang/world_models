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
