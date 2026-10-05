"""Contiguous-window collector for the stochastic navigation benchmark (same behaviour policy as selwm.stochnav.collect)."""
import numpy as np
from .stochnav import StochNav


def collect_windows(variant, n_windows=60_000, K=10, n_envs=500, seed=7, mix_goal=0.4, edge_frac=0.25):
    """Returns P (W,K+1,2), F (W,K+1) bool, A (W,K,2): K contiguous transitions per window.

    Windows start from the same start distribution as `collect` (60% uniform over the arena, 40% the task start
    region) and from a band just outside the hazard (edge_frac), follow the same correlated / goal-directed noisy
    actions, and run K steps with absorbing falls (a fallen agent stays put, flag stays 1)."""
    rng = np.random.default_rng(seed + 1)
    env = StochNav(variant, n_envs, seed)
    n_edge = int(edge_frac * n_windows) if len(env.pit) else 0
    kinds = [('main', n_windows - n_edge), ('edge', n_edge)]
    outs = []
    for kind, tot in kinds:
        done_n = 0
        while done_n < tot:
            n = n_envs
            s, g = env.sample_start_goal(n)
            if kind == 'main':
                s = np.where(rng.random((n, 1)) < 0.6, rng.uniform(0.02, 0.98, (n, 2)), s)
            else:
                r = env.pit[rng.integers(0, len(env.pit), n)]
                px = rng.uniform(r[:, 0] - 0.04, r[:, 2] + 0.04)
                band = rng.uniform(0.0, 0.14, n)
                py = np.where(r[:, 1] <= 0.0, r[:, 3] + band, r[:, 1] - band)
                s = np.clip(np.stack([px, py], 1), 0, 1)
            ok = ~env._in(s, env.pit)                                # a non-fallen start inside the pit is invalid
            env.p = s.astype(np.float64); env.goal = g.astype(np.float64)
            env.fell = np.zeros(n, bool); env.reached = np.zeros(n, bool)
            goaldir = rng.random(n) < mix_goal
            a = rng.uniform(-1, 1, (n, 2))
            P = np.zeros((n, K + 1, 2), np.float32); F = np.zeros((n, K + 1), bool); A = np.zeros((n, K, 2), np.float32)
            P[:, 0] = env.p; F[:, 0] = env.fell
            for t in range(K):
                a = 0.85 * a + 0.55 * rng.normal(size=(n, 2))
                tgt = env.goal - env.p
                tgt = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-8)
                act = np.clip(np.where(goaldir[:, None], 0.7 * tgt + 0.6 * rng.normal(size=(n, 2)), a), -1, 1)
                alive = ~env.fell
                pn, fell = env.transition(env.p, act, rng)
                env.p = np.where(alive[:, None], pn, env.p)
                env.fell |= alive & fell
                A[:, t] = act; P[:, t + 1] = env.p; F[:, t + 1] = env.fell
            P, F, A = P[ok], F[ok], A[ok]
            take = min(len(P), tot - done_n)
            outs.append((P[:take], F[:take], A[:take])); done_n += take
    P = np.concatenate([o[0] for o in outs]); F = np.concatenate([o[1] for o in outs]); A = np.concatenate([o[2] for o in outs])
    perm = rng.permutation(len(P))
    return P[perm], F[perm], A[perm]


def pit_distance(P, F, pit):
    """Chebyshev-ish distance of each position to the pit rectangle(s) (0 inside). P:(...,2)"""
    x, y = P[..., 0:1], P[..., 1:2]
    dx = np.maximum(np.maximum(pit[:, 0] - x, x - pit[:, 2]), 0)
    dy = np.maximum(np.maximum(pit[:, 1] - y, y - pit[:, 3]), 0)
    return np.sqrt(dx ** 2 + dy ** 2).min(-1)
