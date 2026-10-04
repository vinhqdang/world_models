"""KL-tracked CEM and an online estimator of the optimism curve G(KL)."""
import math
import torch


def kl_diag_gauss(mu, sd, sd0):
    """KL( N(mu, sd^2) || N(0, sd0^2) ), summed over all plan dimensions. mu, sd: (B, H, A)."""
    kl = torch.log(sd0 / sd) + (sd ** 2 + mu ** 2) / (2 * sd0 ** 2) - 0.5
    return kl.flatten(1).sum(1)


@torch.no_grad()
def cem_trajectory(cost_fn, B, H, A, N, iters, elite_frac=0.1, init_std=0.8, gen=None, min_std=0.05):
    """CEM that records every iterate.

    cost_fn(acts (B,N,H,A)) -> (B,N) model costs.
    Returns mus (iters+1, B, H, A) [index 0 = prior mean], kls (iters+1, B) w.r.t. the prior.
    """
    mu = torch.zeros(B, H, A)
    sd = torch.full((B, H, A), init_std)
    ne = max(3, int(N * elite_frac))
    mus, kls = [mu.clone()], [kl_diag_gauss(mu, sd, init_std)]
    for _ in range(iters):
        acts = (mu[:, None] + sd[:, None] * torch.randn(B, N, H, A, generator=gen)).clamp(-1, 1)
        c = cost_fn(acts)
        idx = c.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1)
        sd = el.std(1).clamp_min(min_std)
        mus.append(mu.clone())
        kls.append(kl_diag_gauss(mu, sd, init_std))
    return torch.stack(mus), torch.stack(kls)


class OptimismCurve:
    """Online estimate of G(kl) = E[ (J - c_hat) / scale | kl ] by isotonic (PAV) regression.

    Only the outcome of the plan that was actually executed is ever observed.
    """

    def __init__(self, prior_slope=0.0):
        self.kl, self.e = [], []

    def add(self, kl, e):
        self.kl.append(float(kl)); self.e.append(float(e))

    def fit(self):
        import numpy as np
        kl = np.array(self.kl); e = np.array(self.e)
        o = np.argsort(kl); kl, e = kl[o], e[o]
        # pool-adjacent-violators for a non-decreasing fit
        vals, wts, xs = [], [], []
        for k, v in zip(kl, e):
            vals.append(v); wts.append(1.0); xs.append([k])
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                xs[-2] += xs[-1]
                vals[-2], wts[-2] = v2, w
                vals.pop(); wts.pop(); xs.pop()
        self._x = np.concatenate([[np.mean(x)] for x in xs]) if xs else np.array([])
        self._v = np.array(vals)
        return self

    def __call__(self, kl):
        import numpy as np
        if len(self._x) == 0:
            return np.zeros_like(np.asarray(kl, float))
        return np.interp(kl, self._x, self._v)
