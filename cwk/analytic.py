"""Analytic worlds with a known conditional kernel (unit tests, Stage 1 dial).

State s_t = (x_t in R^d, c_t) with exogenous context c_t (AR(1)); action a in [-1, 1]^d (a_lim = 1); outcome
    x_{t+1} = rho x_t + m(c_t) { g a + sum_j sigma_j eps_j d ⊙ f_j(a) },     c_{t+1} = 0.8 c_t + 0.6 eta,
    eps_j = tanh(driver_j), driver_j = sqrt(1 - v) field_j(t) + sqrt(v) n_t (persistent share 1 - v),
    d = (+1, -1, +1, ...), f_j given by the channel kind (cwk.sim.profile; "affine" with g = lambda reproduces
    Y = c{g a + sigma T [(1 - lambda) + lambda u]}, u = a / a_max of the paper's equation (39)).
Unconditionally (v = 1) K(a, a' | c) = m(c)^2 Var(tanh G) sum_j sigma_j^2 (d ⊙ f_j(a)) (d ⊙ f_j(a'))^T with
Var(tanh G) = 0.3943 for G ~ N(0, 1).
"""
import numpy as np

from cwk import upstream  # noqa: F401
from cwk.sim import profile
from cwm.sims.fields import EpisodeField, conditional_driver

VAR_TANH = 0.39430  # Var tanh(G), G ~ N(0, 1)


class Analytic:
    def __init__(self, channels, d=2, gain=0.383, rho=0.5, iid=1.0, seed=0, ctx_gain=0.3, x_scale=0.30, noise=0.0):
        self.channels, self.d, self.gain, self.rho = channels, d, gain, rho
        self.iid, self.seed, self.ctx_gain, self.x_scale, self.noise = float(iid), seed, ctx_gain, x_scale, noise
        self.nd = len(channels)
        self.dirn = np.array([(-1.0) ** i for i in range(d)])
        self.fields = [EpisodeField(seed + 977 * j, [4.0]) for j in range(self.nd)]
        self.D = d + 1
        self.a_lim = 1.0
        self.axes = list(range(d))

    # response ------------------------------------------------------------------------------------------------------
    def m(self, c):
        return 1.0 + self.ctx_gain * np.tanh(c)

    def f(self, j, a):
        ch = self.channels[j]
        kw = {k: v for k, v in ch.items() if k not in ("kind", "scale")}
        kind = ch.get("kind", "affine")
        return profile(kind, a / self.a_lim, **kw)

    def mean_next(self, x, c, a):
        return self.rho * x + self.m(c)[..., None] * self.gain * a

    def outcome(self, x, c, a, eps):
        """x (.., d), c (..,), a (.., d), eps (.., nd) -> next x."""
        pert = sum(self.channels[j]["scale"] * eps[..., j, None] * self.dirn * self.f(j, a) for j in range(self.nd))
        return self.rho * x + self.m(c)[..., None] * (self.gain * a + pert)

    def true_kernel(self, c, a1, a2):
        """K(a1, a2 | c) for v = 1 (unconditional drivers), arrays (.., d) -> (.., d, d)."""
        K = 0
        for j in range(self.nd):
            u1, u2 = self.dirn * self.f(j, a1), self.dirn * self.f(j, a2)
            K = K + self.channels[j]["scale"] ** 2 * VAR_TANH * u1[..., :, None] * u2[..., None, :]
        return self.m(c)[..., None, None] ** 2 * K

    # data ----------------------------------------------------------------------------------------------------------
    def _drivers(self, rng, parent, T):
        """Drivers (T, nd) of one episode."""
        v = self.iid
        out = np.empty((T, self.nd))
        for j in range(self.nd):
            f = np.array([self.fields[j](np.array([float(t)]), parent)[0] for t in range(T)]) if v < 1 else 0.0
            out[:, j] = np.sqrt(1 - v) * f + np.sqrt(v) * rng.standard_normal(T)
        return out

    def collect(self, n, T=5, start=0):
        rng = np.random.default_rng(self.seed + 11)
        S = np.empty((n, T + 1, self.D))
        A = rng.uniform(-1, 1, (n, T, self.d))
        E = np.empty((n, T, self.nd))
        S[:, 0, :self.d] = self.x_scale * rng.standard_normal((n, self.d))
        S[:, 0, self.d] = rng.standard_normal(n)
        for e in range(n):
            E[e] = np.tanh(self._drivers(rng, start + e, T))
        for t in range(T):
            x, c = S[:, t, :self.d], S[:, t, self.d]
            S[:, t + 1, :self.d] = self.outcome(x, c, A[:, t], E[:, t])
            if self.noise:
                S[:, t + 1, :self.d] += self.noise * rng.standard_normal((n, self.d))
            S[:, t + 1, self.d] = 0.8 * c + 0.6 * rng.standard_normal(n)
        return dict(states=S, actions=A, eps=E, iid_fraction=self.iid, action_limit=1.0)

    def population(self, parents, draws, k=4, start=0):
        rng = np.random.default_rng(self.seed + 29)
        n, e = parents, draws
        src = np.empty((n, 3, self.D))
        past_a = rng.uniform(-1, 1, (n, 2, self.d))
        past_eps = np.empty((n, 2, self.nd))
        eps = np.empty((n, e, self.nd))
        post = np.empty((n, self.nd, 2))
        src[:, 0, :self.d] = self.x_scale * rng.standard_normal((n, self.d))
        src[:, 0, self.d] = rng.standard_normal(n)
        basis = [self.fields[0].basis(np.array([float(t)])) for t in range(3)]
        for i in range(n):
            g = np.empty((2, self.nd))
            for j in range(self.nd):
                f = [self.fields[j](np.array([float(t)]), start + i)[0] for t in range(2)] if self.iid < 1 else [0, 0]
                g[:, j] = np.sqrt(1 - self.iid) * np.array(f) + np.sqrt(self.iid) * rng.standard_normal(2)
                mu, var = conditional_driver(np.asarray(basis[:2]), g[:, j], basis[2], self.iid)
                post[i, j] = mu, var
                eps[i, :, j] = np.tanh(mu + np.sqrt(var) * rng.standard_normal(e))
            past_eps[i] = np.tanh(g)
        for t in range(2):
            x, c = src[:, t, :self.d], src[:, t, self.d]
            src[:, t + 1, :self.d] = self.outcome(x, c, past_a[:, t], past_eps[:, t])
            src[:, t + 1, self.d] = 0.8 * c + 0.6 * rng.standard_normal(n)
        actions = rng.uniform(-1, 1, (n, k, self.d))
        x, c = src[:, 2, :self.d], src[:, 2, self.d]
        truth = np.empty((n, e, k, self.D))
        for kk in range(k):
            xn = self.outcome(x[:, None], c[:, None], actions[:, None, kk], eps)
            truth[:, :, kk, :self.d] = xn
        truth[:, :, :, self.d] = (0.8 * c + 0.6 * rng.standard_normal(n))[:, None, None]
        w = np.random.default_rng(self.seed + 5).multinomial(n, np.full(n, 1 / n), size=512)
        return dict(sources=src, past_actions=past_a, past_eps=past_eps, actions=actions, eps=eps, truth=truth,
                    posterior=post, weights=w, iid_fraction=self.iid, action_limit=1.0)


def dial(lam, d=2, iid=1.0, seed=0, sigma=0.609, gain=0.383):
    """Rank-one dial of equation (39): one driver with perturbation sigma eps [(1 - lambda) + lambda u]."""
    return Analytic([dict(kind="affine", g=lam, scale=sigma)], d=d, gain=gain, iid=iid, seed=seed)


def two_driver(nu, d=2, iid=1.0, seed=0, sigma=0.7, gain=0.30):
    """Rank-two analogue: independent offset (weight 1 - nu) and gain (weight nu) drivers."""
    return Analytic([dict(kind="offset", scale=sigma * (1 - nu)), dict(kind="gain", scale=sigma * nu)], d=d,
                    gain=gain, iid=iid, seed=seed)
