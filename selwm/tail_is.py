"""Importance sampling in the explicit noise space of a generative latent world model.

The stochastic predictor is z' = f(z, a, u), u ~ N(0, I). The failure event is 'the imagined latent enters the ball of
radius sqrt(tau) around a failure exemplar at some step'. We tilt the noise proposal q = N(mu, I) in the direction that most
increases a smooth failure score (gradient w.r.t. the noise at u = 0) and correct with exact weights p(u)/q(u).
"""
import torch


def _roll(model, z0, acts, u, anchor_z, tau, stage=None):
    """z0 (B,D), acts (B,H,A), u (B,P,H,nd) -> (min_t dist2 to failure anchor (B,P), final latent (B,P,D))."""
    B, P, H, nd = u.shape
    z = z0[:, None, :].expand(B, P, -1)
    dmin = None
    for t in range(H):
        a = acts[:, None, t, :].expand(B, P, -1)
        z = model.pred(z, a, u[:, :, t])[0]
        d = ((z - anchor_z) ** 2).sum(-1)
        dmin = d if dmin is None else torch.minimum(dmin, d)
    return dmin, z


def tilt_direction(model, z0, acts, anchor_z, tau, eta=2.0, smooth=0.5):
    """Mean shift mu (B,H,nd) of the noise proposal: normalised gradient of the smooth failure score at u = 0."""
    B, H, A = acts.shape
    nd = model.noise_dim
    u0 = torch.zeros(B, 1, H, nd, requires_grad=True)
    z = z0[:, None, :]
    ds = []
    with torch.enable_grad():
        for t in range(H):
            z = model.pred(z, acts[:, None, t, :], u0[:, :, t])[0]
            ds.append(((z - anchor_z) ** 2).sum(-1))
        d = torch.stack(ds, -1)                                          # (B,1,H)
        score = -smooth * torch.logsumexp(-d / smooth, dim=-1)           # smooth -min_t d  (higher = closer to failure)
        g, = torch.autograd.grad(score.sum(), u0)
    g = g[:, 0]                                                          # (B,H,nd)
    nrm = g.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
    return eta * g / nrm


@torch.no_grad()
def fail_prob(model, z0, acts, anchor, M, mode='mc', mu=None, mix=0.5, gen=None):
    """Estimate P(failure) per candidate. mode in {'mc', 'is', 'mix'}. z0 (B,D), acts (B,H,A) -> (B,) estimates."""
    B, H, A = acts.shape
    nd = model.noise_dim
    eps = torch.randn(B, M, H, nd, generator=gen)
    if mode == 'mc':
        u, w = eps, torch.ones(B, M)
    else:
        mu_b = mu[:, None]                                               # (B,1,H,nd)
        if mode == 'is':
            u = mu_b + eps
            w = torch.exp(-(u * mu_b).flatten(2).sum(-1) + 0.5 * (mu_b ** 2).flatten(2).sum(-1))
        else:                                                            # defensive mixture (1-mix) p + mix q
            use_q = (torch.rand(B, M, 1, 1, generator=gen) < mix).float()
            u = eps + use_q * mu_b
            logp = -0.5 * (u ** 2).flatten(2).sum(-1)
            logq = -0.5 * ((u - mu_b) ** 2).flatten(2).sum(-1)
            w = torch.exp(logp - torch.logaddexp(logp + torch.log(torch.tensor(1 - mix)), logq + torch.log(torch.tensor(mix))))
    dmin, _ = _roll(model, z0, acts, u, anchor.z, anchor.tau)
    fail = (dmin < anchor.tau).float()
    return (w * fail).mean(1)


@torch.no_grad()
def ce_direction(model, z0, acts, anchor, Mp, damp=1.0, gen=None):
    """Cross-entropy style tilt: mean of the pilot noise over particles that failed (damped); zero if none failed."""
    B, H, A = acts.shape
    eps = torch.randn(B, Mp, H, model.noise_dim, generator=gen)
    dmin, _ = _roll(model, z0, acts, eps, anchor.z, anchor.tau)
    f = (dmin < anchor.tau).float()[..., None, None]
    n = f.sum(1)
    return damp * (eps * f).sum(1) / n.clamp_min(1.0) * (n > 0).float()
