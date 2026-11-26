"""Glue between populations and predictors: evaluation inputs, broadcast predictions."""
import numpy as np


def pop_inputs(pop):
    """H (N, 3D), s (N, D) anchor context and actions (N, K, p)."""
    src = pop["sources"]
    H = np.concatenate([src[:, 0], src[:, 1], src[:, 2]], -1)
    return H, src[:, 2], pop["actions"]


def mean_predictions(model, pop):
    """mu(H, a_k) for every parent and query action: (N, K, D). Outcome-blind."""
    H, s, A = pop_inputs(pop)
    N, K, p = A.shape
    out = model.predict(np.repeat(H, K, 0), A.reshape(N * K, p), np.repeat(s, K, 0))
    return out.reshape(N, K, -1)


def broadcast(mu, E):
    """(N, K, D) -> (N, E, K, D)."""
    return np.repeat(mu[:, None], E, 1)


def residual_reuse(mu, truth):
    """y_hat(a') = y(a_0) + mu(a') - mu(a_0) for every draw: (N, E, K, D)."""
    y0 = truth[:, :, :1]
    return y0 + mu[:, None] - mu[:, None, :1]
