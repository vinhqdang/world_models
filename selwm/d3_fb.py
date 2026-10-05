"""Feedback-aware CEM: plan = nominal actions a^nom_t plus a diagonal feedback gain K_t on the deviation of the sampled
state from the noise-free nominal trajectory,  a_t = clip(a^nom_t + K_t * (s_t - sbar_t)).  (a^nom, K) are optimised jointly
by CEM against noisy particles; only a^nom_0 is executed (the deviation is zero at t=0), then the agent replans."""
import torch
from .risk_plan import OracleModel


@torch.no_grad()
def fb_cem(model, obs0, goal, H, N, M, iters=3, elite_frac=0.1, init_std=0.7, gen=None, score_fn=None,
           kmode='const', kmax=15.0, k_init=0.5, k_sd=0.35, use_fb=True):
    """model.rollout_fb(obs0, acts (E,N,H,A), K (E,N,KH,A), M, gen, goal) -> features; model.cost(feat, goal) -> (E,N,M).
    kmode 'const': one gain per axis shared over the horizon; 'step': one gain per axis and time step. K in [-kmax, 0]."""
    E, A = obs0.shape[0], 2
    KH = H if kmode == 'step' else 1
    dv = obs0.device
    mu = torch.zeros(E, 1, H, A, device=dv)
    sd = torch.full((E, 1, H, A), init_std, device=dv)
    kmu = torch.full((E, 1, KH, A), -k_init * kmax, device=dv)
    ksd = torch.full((E, 1, KH, A), k_sd * kmax, device=dv)
    ne = max(3, int(N * elite_frac))
    for _ in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, device=dv, generator=gen)).clamp(-1, 1)
        K = (kmu + ksd * torch.randn(E, N, KH, A, device=dv, generator=gen)).clamp(-kmax, 0.0)
        if not use_fb:
            K = torch.zeros_like(K)
        feat = model.rollout_fb(obs0, acts, K, M, gen, goal)
        c = model.cost(feat, goal)
        s = score_fn(c) if score_fn is not None else c.mean(-1)
        idx = s.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1, keepdim=True)
        sd = el.std(1, keepdim=True).clamp_min(0.05)
        if use_fb:
            ek = torch.gather(K, 1, idx[:, :, None, None].expand(-1, -1, KH, A))
            kmu = ek.mean(1, keepdim=True)
            ksd = ek.std(1, keepdim=True).clamp_min(0.02 * kmax)
    return mu[:, 0], kmu[:, 0]


class OracleFB(OracleModel):
    """True simulator with the feedback law on the position deviation from the noise-free nominal path."""

    def rollout_fb(self, p0, acts, K, M, gen, goal=None):
        E, N, H, _ = acts.shape
        Me = 1 if self.det else M
        acc = torch.zeros(E, N, Me, device=p0.device)
        p = p0[:, None, None, :].expand(E, N, Me, 2).clone()
        pbar = p0[:, None, None, :].expand(E, N, 1, 2).clone()
        stuck = torch.zeros(E, N, Me, dtype=torch.bool, device=p0.device)
        for t in range(H):
            an = acts[:, :, t, None, :].clamp(-1, 1)
            Kt = K[:, :, min(t, K.shape[2] - 1), None, :]
            a = (an + Kt * (p - pbar)).clamp(-1, 1)
            pn = p + self.step * a
            if not self.det:
                sign = (torch.randint(0, 2, (E, N, Me), device=p0.device, generator=gen) * 2 - 1).float()
                pn[..., 1] = pn[..., 1] + self.wind * sign
            pn = pn.clamp(0, 1)
            pn = torch.where(stuck[..., None], p, pn)
            stuck = stuck | self._in_pit(pn)
            p = pn
            pbar = (pbar + self.step * an).clamp(0, 1)
            if self.stage_w > 0:
                d = ((p - goal[:, None, None, :]) ** 2).sum(-1)
                acc = acc + torch.where(stuck, torch.full_like(d, self.FAIL_COST), d) / H
        out = torch.cat([p, stuck[..., None].float()], -1)
        return torch.cat([out, acc[..., None]], -1) if self.stage_w > 0 else out
