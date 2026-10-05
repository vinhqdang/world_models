"""Scenario / candidate generators shared by n2_rank.py and n2_learn_nodes.py (copied verbatim from n2_rank.py)."""
import numpy as np, torch
from selwm.stochnav import StochNav


def scenarios(S, rng, variant='cliff_hi'):
    env = StochNav(variant, 1, int(rng.integers(1 << 30)))
    _, g = env.sample_start_goal(S)
    p = np.stack([rng.uniform(0.08, 0.80, S), rng.uniform(0.30, 0.50, S)], 1)
    return torch.tensor(p, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)


def candidates(p, g, N, H, regime, rng_t):
    S = p.shape[0]
    d = g - p; d = d / d.norm(dim=-1, keepdim=True)
    if regime == 'wide':                                               # d6 regime: large lateral spread, easy global ordering
        yoff = (torch.rand(S, N, 1, generator=rng_t) * 1.6 - 0.8)
        eps = 0.4 * torch.randn(S, N, H, 2, generator=rng_t)
    else:                                                              # late-CEM regime: local set around a plan, hard local ordering
        yoff = (torch.rand(S, 1, 1, generator=rng_t) * 0.8 - 0.1) + 0.12 * torch.randn(S, N, 1, generator=rng_t)
        eps = 0.15 * torch.randn(S, N, H, 2, generator=rng_t)
    act = d[:, None, None, :] + eps
    act[..., 1] = act[..., 1] + yoff
    return act.clamp(-1, 1)


