"""Counterfactual R^2 with whole-parent bootstrap, following cwm/metrics.py of the authors' repository (MIT licence).

Population layout: (parents N, draws E, actions K, coordinates D); action 0 is the executed action and only the
alternatives (1:) are scored. Everything is built from per-parent losses so that paired differences between methods use
the same bootstrap resamples (stored parent multiplicities ``weights`` (B, N) and a seed-resampling matrix).
"""
import numpy as np


def _q(x):
    return np.quantile(x, [.025, .975]).tolist()


def target_variance(truth, coords):
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    return float(np.mean(np.sum((y - y.mean((0, 1, 2))) ** 2, -1)))


def bootstrap_denominator(truth, coords, weights):
    """Per-resample outcome variance, as in cwm.metrics.scores."""
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    n, e, a, _ = y.shape
    m = e * a
    w = np.asarray(weights, np.float64)
    return (w @ (y * y).sum((1, 2, 3))) / (n * m) - np.sum(((w @ y.sum((1, 2))) / (n * m)) ** 2, axis=1)


def parent_loss(pred, truth, coords):
    """Mean over draws and alternative actions of the summed squared error: (N,). ``pred``: (N, E, K, D)."""
    p = np.asarray(pred, np.float64)[:, :, 1:][..., coords]
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    return np.mean(np.sum((p - y) ** 2, -1), axis=(1, 2))


def parent_action_loss(pred, truth, coords):
    """Per (parent, alternative action) loss, mean over draws: (N, K - 1)."""
    p = np.asarray(pred, np.float64)[:, :, 1:][..., coords]
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    return np.mean(np.sum((p - y) ** 2, -1), axis=1)


def oracle_loss(truth, coords):
    """Outcome-blind oracle: unbiased within-parent variance (the exact simulator mean has no estimation error)."""
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    return y.var(axis=1, ddof=1).sum(-1).mean(-1)


def exact_mean_residual_loss(truth, coords):
    """Residual reuse with exact within-parent means: unbiased variance of the individual action contrast."""
    y = np.asarray(truth, np.float64)[..., coords]
    eff = y[:, :, 1:] - y[:, :, :1]
    return eff.var(axis=1, ddof=1).sum(-1).mean(-1)


class Evaluator:
    """R^2 and bootstrap quantities for a fixed population and coordinate group."""

    def __init__(self, truth, weights, coords, n_seeds=5, seed=517):
        self.truth, self.coords = np.asarray(truth), list(coords)
        self.var = target_variance(truth, self.coords)
        self.den = bootstrap_denominator(truth, self.coords, weights)
        self.w = np.asarray(weights, np.float64)
        self.n = self.truth.shape[0]
        b = len(self.w)
        self.sw = np.random.default_rng(seed).multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=b) / n_seeds
        self.n_seeds = n_seeds

    def loss(self, pred):
        return parent_loss(pred, self.truth, self.coords)

    def _boot_loss(self, losses):
        """losses (S, N) -> (B,) bootstrap mean losses, resampling seeds (if S > 1) and parents."""
        L = np.atleast_2d(losses)
        s = L.shape[0]
        sw = self.sw if s == self.n_seeds else (np.ones((len(self.w), 1)) if s == 1 else None)
        if sw is None:
            raise ValueError("seed count mismatch")
        return np.einsum("bs,sn,bn->b", sw, L, self.w) / self.n

    def r2(self, losses):
        L = np.atleast_2d(losses)
        boot = 1 - self._boot_loss(L) / self.den
        return dict(r2=float(1 - L.mean() / self.var), ci95=_q(boot),
                    per_seed=(1 - L.mean(1) / self.var).tolist() if L.shape[0] > 1 else None)

    def diff(self, loss_a, loss_b):
        """R^2(a) - R^2(b) with paired bootstrap interval (positive: a better)."""
        La, Lb = np.atleast_2d(loss_a), np.atleast_2d(loss_b)
        d = (self._boot_loss(Lb) - self._boot_loss(La)) / self.den
        point = float((Lb.mean() - La.mean()) / self.var)
        lo, hi = _q(d)
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        return dict(diff=point, ci95=[lo, hi], p_boot=max(p, 1.0 / len(d)))


def holm(pvals, alpha=0.05):
    """Holm step-down: returns (reject flags, adjusted p-values) for a list of p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return [bool(a <= alpha) for a in adj], adj.tolist()


def interaction_ratio(truth, coords):
    """rho = mean within-parent variance of the action contrast over mean within-parent outcome variance."""
    y = np.asarray(truth, np.float64)[..., coords]
    cf = y[:, :, 1:]
    noise = cf.var(1, ddof=1).sum(-1).mean(-1)
    effect = (cf - y[:, :, :1]).var(1, ddof=1).sum(-1).mean(-1)
    return float(effect.mean() / noise.mean())
