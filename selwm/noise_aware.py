"""Noise-aware candidate selection, particle racing and failure-rate control for sample-based planners.

Setting: each candidate action sequence i is scored from M sampled outcomes by a risk functional rho_hat_i. These
scores are noisy, and the noise level differs strongly between candidates (candidates that pass close to a hazard
have heavy-tailed costs). Selecting by the raw minimum therefore (i) is optimistic and (ii) favours high-variance
candidates, i.e. it is risk-seeking by accident. We rank by an empirical-Bayes posterior mean instead.
"""
import math
import torch


def mix_score(costs, lam=0.0, tail=0.1):
    """(1 - lam) * E[c] + lam * CVaR_tail[c];  costs (..., M) -> (...). A coherent risk measure for lam in [0, 1]."""
    m = costs.mean(-1)
    if lam == 0:
        return m
    k = max(1, int(math.ceil(tail * costs.shape[-1])))
    return (1 - lam) * m + lam * costs.topk(k, dim=-1).values.mean(-1)


def bootstrap_var(costs, score_fn, B=10, gen=None):
    """Variance of score_fn over particle resamples. costs (E,N,M) -> (E,N). Shared resample indices across candidates."""
    E, N, M = costs.shape
    idx = torch.randint(0, M, (B, M), device=costs.device, generator=gen)
    s = torch.stack([score_fn(costs[..., idx[b]]) for b in range(B)], 0)       # (B,E,N)
    return s.var(0, unbiased=True)


def eb_shrink(score, var, eps=1e-8):
    """Heteroscedastic empirical-Bayes shrinkage towards the pool mean (method-of-moments prior variance).

    score, var: (E, N). Prior variance tau^2 = max(Var_i(score) - mean_i(var), eps). Posterior mean
    m + tau^2 / (tau^2 + var_i) * (score_i - m).
    """
    m = score.mean(1, keepdim=True)
    tau2 = (score.var(1, keepdim=True, unbiased=True) - var.mean(1, keepdim=True)).clamp_min(eps)
    w = tau2 / (tau2 + var)
    return m + w * (score - m)


class NoiseAwareScorer:
    """Callable passed to risk_plan.cem(score_fn=...)."""

    def __init__(self, lam=0.0, tail=0.1, shrink=True, B=10):
        self.lam, self.tail, self.shrink, self.B = lam, tail, shrink, B

    def __call__(self, costs):
        f = lambda c: mix_score(c, self.lam, self.tail)
        s = f(costs)
        if not self.shrink:
            return s
        v = bootstrap_var(costs, f, self.B)
        return eb_shrink(s, v)


@torch.no_grad()
def raced_costs(model, z0, zg, acts, M_stages, keep_fracs, gen=None):
    """Successive-halving particle allocation.

    Stage 0 scores all N candidates with M_stages[0] particles; each later stage keeps the best `keep_fracs[s-1]`
    fraction (by the running mean cost) and adds M_stages[s] fresh independent particles to the survivors.
    Returns costs (E,N,Mmax) padded with NaN where a candidate received fewer particles, and a survivor mask (E,N).
    """
    E, N = acts.shape[:2]
    c0 = model.cost(model.rollout(z0, acts, M_stages[0], gen, zg), zg)                  # (E,N,M0)
    Mmax = sum(M_stages)
    costs = torch.full((E, N, Mmax), float('nan'), device=acts.device)
    costs[..., : M_stages[0]] = c0
    alive = torch.ones(E, N, dtype=torch.bool, device=acts.device)
    used = M_stages[0]
    cur_idx = torch.arange(N, device=acts.device).expand(E, N)
    for s in range(1, len(M_stages)):
        k = max(1, int(N * keep_fracs[s - 1]))
        run = torch.nanmean(costs, -1)
        run = run.masked_fill(~alive, float('inf'))
        top = run.topk(k, dim=1, largest=False).indices                              # (E,k)
        a_k = torch.gather(acts, 1, top[:, :, None, None].expand(-1, -1, acts.shape[2], acts.shape[3]))
        c_new = model.cost(model.rollout(z0, a_k, M_stages[s], gen, zg), zg)            # (E,k,Ms)
        alive = torch.zeros_like(alive).scatter_(1, top, True)
        for j in range(c_new.shape[-1]):
            costs[:, :, used + j].scatter_(1, top, c_new[..., j])
        used += M_stages[s]
    return costs, alive


class RiskController:
    """Online control of the conservativeness lam in [0,1] so that the long-run failure rate stays near delta.

    After every finished episode:  lam <- clip(lam + eta * (failed - delta), 0, 1).
    (Conformal-decision-theory style update; if the failure probability is non-increasing in lam and lam=1 is safe,
    the running failure frequency is within O(1/(eta T)) of delta.)
    """

    def __init__(self, delta=0.05, eta=0.05, lam0=0.3):
        self.delta, self.eta, self.lam = delta, eta, lam0
        self.hist = []

    def update(self, failed_flags):
        for f in failed_flags:
            self.lam = min(1.0, max(0.0, self.lam + self.eta * (float(f) - self.delta)))
            self.hist.append(self.lam)
        return self.lam


@torch.no_grad()
def cem_raced(model, z0, zg, H, N, M_stages, keep_fracs, iters=4, elite_frac=0.1, lam=0.0, tail=0.1, shrink=True,
              init_std=0.7, gen=None):
    """CEM whose candidates are scored with successive-halving particle racing; elites come from the survivors."""
    E = z0.shape[0]; A = 2
    mu = torch.zeros(E, 1, H, A, device=z0.device)
    sd = torch.full((E, 1, H, A), init_std, device=z0.device)
    ne = max(3, int(N * elite_frac))
    scorer = NoiseAwareScorer(lam, tail, shrink)
    k_last = max(ne, int(N * keep_fracs[-1]))
    kf = list(keep_fracs[:-1]) + [k_last / N]
    for _ in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, device=z0.device, generator=gen)).clamp(-1, 1)
        costs, alive = raced_costs(model, z0, zg, acts, M_stages, kf, gen)
        sidx = alive.float().topk(k_last, 1).indices                                  # survivors (E,k_last)
        cs = torch.gather(costs, 1, sidx[:, :, None].expand(-1, -1, costs.shape[-1]))  # (E,k,Mmax) all particles present
        s = scorer(cs)                                                                  # (E,k)
        top = s.topk(ne, dim=1, largest=False).indices
        el_idx = torch.gather(sidx, 1, top)
        el = torch.gather(acts, 1, el_idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1, keepdim=True)
        sd = el.std(1, keepdim=True).clamp_min(0.05)
    return mu[:, 0]
