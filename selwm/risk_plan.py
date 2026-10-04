"""Risk-aware CEM planning over stochastic world models (batched over environments)."""
import math
import torch


def risk_score(costs, risk='expected', alpha=0.25):
    """costs: (E, N, M) per-particle costs (lower is better) -> (E, N) scalar score to minimise."""
    if risk in ('mean', 'expected'):
        return costs.mean(-1)
    if risk == 'cvar':
        k = max(1, int(math.ceil(alpha * costs.shape[-1])))
        return costs.topk(k, dim=-1).values.mean(-1)           # mean of the worst alpha-fraction
    if risk == 'worst':
        return costs.max(-1).values
    raise ValueError(risk)


@torch.no_grad()
def cem(model, obs0, goal, H, N, M, iters=5, elite_frac=0.1, risk='expected', alpha=0.25,
        init_std=0.7, gen=None, score_fn=None, warm=None):
    """Cross-entropy-method planner.

    model.rollout(obs0, acts (E,N,H,A), M, gen) -> final-state features (E,N,M,D)
    model.cost(feat (E,N,M,D), goal) -> (E,N,M)
    score_fn(costs (E,N,M)) -> (E,N)  overrides the risk reduction (used by noise-aware selection).
    Returns the mean action sequence (E,H,A) of the final iteration.
    """
    E = obs0.shape[0]
    A = 2
    mu = torch.zeros(E, 1, H, A, device=obs0.device) if warm is None else warm.unsqueeze(1).clone()
    sd = torch.full((E, 1, H, A), init_std, device=obs0.device)
    ne = max(3, int(N * elite_frac))
    for _ in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, device=obs0.device, generator=gen)).clamp(-1, 1)
        feat = model.rollout(obs0, acts, M, gen)
        c = model.cost(feat, goal)                              # (E,N,M)
        s = score_fn(c) if score_fn is not None else risk_score(c, risk, alpha)
        idx = s.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1, keepdim=True)
        sd = el.std(1, keepdim=True).clamp_min(0.05)
    return mu[:, 0]


class OracleModel:
    """Ground-truth stochastic simulator in state space (upper bound / benchmark sanity check)."""

    def __init__(self, variant='cliff', deterministic=False, step=0.05, wind=0.06):
        from .stochnav import StochNav
        e = StochNav(variant, 1)
        self.pit = torch.tensor(e.pit, dtype=torch.float32)
        self.step, self.wind, self.det = step, wind, deterministic
        self.variant = variant

    def _in_pit(self, p):
        if self.pit.numel() == 0:
            return torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)
        r = self.pit.to(p.device)
        x, y = p[..., 0:1], p[..., 1:2]
        return ((x >= r[:, 0]) & (x <= r[:, 2]) & (y >= r[:, 1]) & (y <= r[:, 3])).any(-1)

    def rollout(self, p0, acts, M, gen):
        E, N, H, _ = acts.shape
        M_eff = 1 if self.det else M
        p = p0[:, None, None, :].expand(E, N, M_eff, 2).clone()
        stuck = torch.zeros(E, N, M_eff, dtype=torch.bool, device=p0.device)
        for t in range(H):
            a = acts[:, :, t, None, :].clamp(-1, 1)
            pn = p + self.step * a
            if not self.det:
                sign = (torch.randint(0, 2, (E, N, M_eff), device=p0.device, generator=gen) * 2 - 1).float()
                pn[..., 1] = pn[..., 1] + self.wind * sign
            pn = pn.clamp(0, 1)
            pn = torch.where(stuck[..., None], p, pn)
            stuck = stuck | self._in_pit(pn)
            p = pn
        return torch.cat([p, stuck[..., None].float()], -1)

    FAIL_COST = 1.5

    def cost(self, feat, goal):
        c = ((feat[..., :2] - goal[:, None, None, :]) ** 2).sum(-1)
        return torch.where(feat[..., 2] > 0.5, torch.full_like(c, self.FAIL_COST), c)
