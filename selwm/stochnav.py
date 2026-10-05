"""Vectorised stochastic navigation benchmark with pixel observations.

Two variants share one engine:

  cliff : uniform bimodal wind (+/- delta along y with equal probability) and an absorbing pit placed on
          the straight line between start and goal. A deterministic (mean) model sees zero net drift and
          hugs the pit edge; the true dynamics push the agent into the pit about half of the time.
  gust  : no pit; a windy band across the direct path has a much larger wind amplitude than the rest of the
          arena, so the expected squared latent distance of a path depends on its variance.

Observations are 3 x S x S images that are a deterministic function of the agent position, so they are rendered on
the fly with torch (`render`) for both training and evaluation. Ground-truth positions are only used for scoring.
"""
import numpy as np
import torch

SIZE = 48


class StochNav:
    def __init__(self, variant='cliff', n_envs=1, seed=0, step=0.05, wind=None, goal_r=0.06, max_steps=120):
        self.variant, self.E, self.step_size, self.goal_r, self.max_steps = variant, n_envs, step, goal_r, max_steps
        self.rng = np.random.default_rng(seed)
        if wind is None:
            wind = {'bridge': 0.035, 'cliff_hi': 0.09}.get(variant, 0.06)
        if variant in ('cliff', 'cliff_hi'):
            self.pit = np.array([[0.25, 0.0, 0.75, 0.30]])          # x0,y0,x1,y1
            self.gust = np.zeros((0, 4))
            self.wind_base, self.wind_gust = wind, wind
        elif variant == 'bridge':
            self.pit = np.array([[0.30, 0.0, 0.70, 0.28], [0.30, 0.72, 0.70, 1.0]])
            self.gust = np.zeros((0, 4))
            self.wind_base, self.wind_gust = wind, wind
        elif variant == 'gust':
            self.pit = np.zeros((0, 4))
            self.gust = np.array([[0.30, 0.0, 0.70, 0.45]])
            self.wind_base, self.wind_gust = 0.015, 0.12
        else:
            raise ValueError(variant)
        self.reset()

    # ---------------------------------------------------------------- sampling
    def sample_start_goal(self, n):
        r = self.rng
        if self.variant == 'bridge':          # start and goal both near the same (random) edge of the corridor
            side = np.where(r.random(n) < 0.5, -1.0, 1.0)
            ys_ = 0.5 + side * r.uniform(0.15, 0.20, n)
            yg_ = 0.5 + side * r.uniform(0.15, 0.20, n)
        else:
            ys_, yg_ = r.uniform(0.34, 0.46, n), r.uniform(0.34, 0.46, n)
        s = np.stack([r.uniform(0.06, 0.18, n), ys_], 1)
        g = np.stack([r.uniform(0.82, 0.94, n), yg_], 1)
        return s, g

    def reset(self, start=None, goal=None):
        if start is None:
            start, goal = self.sample_start_goal(self.E)
        self.p, self.goal = start.copy().astype(np.float64), goal.copy().astype(np.float64)
        self.fell = np.zeros(self.E, bool)
        self.reached = np.zeros(self.E, bool)
        self.t = 0
        return self.p.copy()

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def _in(p, rects):
        if len(rects) == 0:
            return np.zeros(p.shape[:-1], bool)
        x, y = p[..., 0:1], p[..., 1:2]
        return ((x >= rects[:, 0]) & (x <= rects[:, 2]) & (y >= rects[:, 1]) & (y <= rects[:, 3])).any(-1)

    def wind_amp(self, p):
        return np.where(self._in(p, self.gust), self.wind_gust, self.wind_base)

    # ---------------------------------------------------------------- dynamics
    def transition(self, p, a, rng):
        """One stochastic step for positions p (N,2) and actions a (N,2); returns p', fell."""
        a = np.clip(a, -1, 1)
        pn = p + self.step_size * a
        sign = rng.choice([-1.0, 1.0], size=len(p))
        pn[:, 1] += self.wind_amp(p) * sign
        pn = np.clip(pn, 0.0, 1.0)
        fell = self._in(pn, self.pit)
        return pn, fell

    def step(self, a):
        alive = ~(self.fell | self.reached)
        pn, fell = self.transition(self.p, a, self.rng)
        self.p = np.where(alive[:, None], pn, self.p)
        self.fell |= alive & fell
        d = np.linalg.norm(self.p - self.goal, axis=1)
        self.reached |= alive & (~self.fell) & (d < self.goal_r)
        self.t += 1
        return self.p.copy(), self.fell.copy(), self.reached.copy()


# -------------------------------------------------------------------------- rendering
def _pit_mask(variant, device):
    ys, xs = torch.meshgrid(torch.linspace(0, 1, SIZE, device=device), torch.linspace(0, 1, SIZE, device=device), indexing='ij')
    m = torch.zeros(SIZE, SIZE, device=device)
    if variant in ('cliff', 'cliff_hi'):
        m = ((xs >= 0.25) & (xs <= 0.75) & (ys <= 0.30)).float()
    elif variant == 'bridge':
        m = ((xs >= 0.30) & (xs <= 0.70) & ((ys <= 0.28) | (ys >= 0.72))).float()
    return m, ys, xs


def render(p, variant='cliff', fell=None, sigma_px=1.6):
    """p: (...,2) tensor in [0,1]^2 (x right, y up) -> (..., 3, S, S) float images in [0,1].

    If `fell` (bool, shape p.shape[:-1]) is True the observation is a dark 'game over' frame.
    """
    device = p.device
    pit, ys, xs = _pit_mask(variant, device)
    lead = p.shape[:-1]
    pf = p.reshape(-1, 2)
    px = pf[:, 0, None, None] * (SIZE - 1)
    py = pf[:, 1, None, None] * (SIZE - 1)
    gx = xs[None] * (SIZE - 1)
    gy = ys[None] * (SIZE - 1)
    blob = torch.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * sigma_px ** 2))          # (N,S,S)
    base = torch.full((pf.shape[0], 3, SIZE, SIZE), 0.92, device=device)
    pitc = torch.tensor([0.08, 0.08, 0.35], device=device).view(1, 3, 1, 1)
    img = base * (1 - pit[None, None]) + pitc * pit[None, None]
    if variant == 'gust':
        gm = ((xs >= 0.30) & (xs <= 0.70) & (ys <= 0.45)).float()[None, None]
        img = img * (1 - 0.25 * gm) + torch.tensor([0.55, 0.75, 0.95], device=device).view(1, 3, 1, 1) * 0.25 * gm
    red = torch.tensor([0.95, 0.1, 0.1], device=device).view(1, 3, 1, 1)
    img = img * (1 - blob[:, None]) + red * blob[:, None]
    img = torch.flip(img, dims=[2])                                                      # y up -> image rows down
    if fell is not None:
        f = fell.reshape(-1).to(img.dtype)[:, None, None, None]
        img = img * (1 - f) + 0.0 * f
    return img.reshape(*lead, 3, SIZE, SIZE)


def collect(variant, n_steps_total=200_000, n_envs=500, seed=0, mix_goal=0.4, edge_frac=0.25):
    """Behaviour data: correlated random actions, plus noisy goal-directed segments so edge regions are covered.

    Returns arrays (p_t, a_t, p_{t+1}) with episodes restarted when an agent falls or reaches the goal.
    """
    n_edge = int(edge_frac * n_steps_total)
    n_steps_total = n_steps_total - n_edge
    env = StochNav(variant, n_envs, seed)
    rng = np.random.default_rng(seed + 1)
    T = n_steps_total // n_envs
    P = np.zeros((T, n_envs, 2), np.float32); A = np.zeros((T, n_envs, 2), np.float32); P2 = np.zeros_like(P)
    F = np.zeros((T, n_envs), bool); F2 = np.zeros((T, n_envs), bool)
    a = rng.uniform(-1, 1, (n_envs, 2))
    goaldir = rng.random(n_envs) < mix_goal
    for t in range(T):
        # Ornstein-Uhlenbeck style correlated actions
        a = 0.85 * a + 0.55 * rng.normal(size=(n_envs, 2))
        tgt = env.goal - env.p
        tgt = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-8)
        act = np.where(goaldir[:, None], 0.7 * tgt + 0.6 * rng.normal(size=(n_envs, 2)), a)
        act = np.clip(act, -1, 1)
        P[t] = env.p; F[t] = env.fell
        alive = ~(env.fell | env.reached)
        pn, fell = env.transition(env.p, act, rng)
        A[t] = act
        P2[t] = np.where(alive[:, None], pn, env.p)
        env.p = P2[t].astype(np.float64)
        env.fell |= alive & fell
        F2[t] = env.fell
        d = np.linalg.norm(env.p - env.goal, axis=1)
        env.reached |= alive & (~env.fell) & (d < env.goal_r)
        done = env.fell | env.reached | (rng.random(n_envs) < 0.01)
        if done.any():
            s, g = env.sample_start_goal(int(done.sum()))
            # restart anywhere in the arena (not only the left side) so all of the state space is covered
            s = np.where(rng.random((len(s), 1)) < 0.6, rng.uniform(0.02, 0.98, (len(s), 2)), s)
            env.p[done] = s; env.goal[done] = g
            env.fell[done] = False; env.reached[done] = False
            goaldir[done] = rng.random(int(done.sum())) < mix_goal
    keep = ~env._in(P.reshape(-1, 2).astype(np.float64), env.pit) | F.reshape(-1)     # a pit position is only valid as a fallen state
    out = [P.reshape(-1, 2)[keep], F.reshape(-1)[keep], A.reshape(-1, 2)[keep], P2.reshape(-1, 2)[keep], F2.reshape(-1)[keep]]
    if n_edge > 0 and len(env.pit):
        # i.i.d. one-step transitions from a band just outside the hazard (deliberate data collection near the risk)
        r = env.pit[rng.integers(0, len(env.pit), n_edge)]
        px = rng.uniform(r[:, 0] - 0.04, r[:, 2] + 0.04)
        band = rng.uniform(0.0, 0.14, n_edge)
        py = np.where(r[:, 1] <= 0.0, r[:, 3] + band, r[:, 1] - band)
        pe = np.clip(np.stack([px, py], 1), 0, 1)
        pe = pe[~env._in(pe, env.pit)]
        ae = rng.uniform(-1, 1, (len(pe), 2))
        pn, fe = env.transition(pe, ae, rng)
        out = [np.concatenate([o, e]) for o, e in zip(out, [pe.astype(np.float32), np.zeros(len(pe), bool), ae.astype(np.float32), pn.astype(np.float32), fe])]
    perm = rng.permutation(len(out[0]))
    return tuple(o[perm] for o in out)
