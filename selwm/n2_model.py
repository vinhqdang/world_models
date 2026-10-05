"""N2: deterministic / structured alternatives to M iid particles for the energy-score latent predictor.

(1) Active-direction analysis.  For the trained es predictor z' = z + f(z, a, u), u ~ N(0, I_8), the Jacobian df/du has one
    dominant singular value (0.26-0.28 against 0.01 and 0.003) and the response along that direction is a saturated sigmoid:
    the learned one-step law is a two-atom mixture.  `active_direction` extracts the common direction vbar by averaging the
    oriented top right-singular vectors over random (z, a); `atom_weight` estimates P(+) by Monte Carlo over random (z, a).
    Both use random states in the arena only (no evaluation data).

(2) Node designs along vbar (all shared across candidates, i.e. coupled / CRN):
      dir_iid, dir_lhs, dir_lhs_anti : iid / Latin-hypercube / antithetic 1-D coordinates q[m, t]; u = q vbar + xi (xi iid, shared)
      dir_oa                         : +-c sign design with (near-)orthogonal balanced columns (supersaturated-design search),
                                       randomised by independent column sign flips and a random column-to-step assignment
      learned                        : node tensor U[m, t, :] trained offline to minimise ranking error (see experiments/n2_learn_nodes.py)

(3) AtomLatentModel: deterministic belief propagation.  The belief is a weighted atom cloud; every step each atom is branched
    into the two atoms f(z, a, +c vbar), f(z, a, -c vbar) with weights w * p, w * (1-p), the accumulators (stage cost, failure
    flag with noisy-or update) are carried per atom, and clouds with more than K atoms are compressed by repeatedly merging the
    pair with the smallest Ward cost w_i w_j / (w_i + w_j) ||z_i - z_j||^2 (barycentre, weighted accumulators).  No random numbers.
"""
import math
import numpy as np
import torch
from .jepa import LatentModel
from .d6_model import CoupledLatentModel
from .d6_noise import NoiseSource, _ndtri, _rand


@torch.no_grad()
def _random_states(n, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.stack([torch.rand(n, generator=g) * 0.9 + 0.05, torch.rand(n, generator=g) * 0.9 + 0.05], 1)
    z = torch.cat([p * 2 - 1, torch.zeros(n, 1)], 1)
    a = torch.rand(n, 2, generator=g) * 2 - 1
    return z, a


def active_direction(jepa, n=4000, seed=12345):
    """Common dominant noise direction of the learned predictor (unit vector in R^K), oriented so that +vbar raises z_y."""
    P = jepa.pred
    z, a = _random_states(n, seed)
    u = torch.zeros(n, P.noise_dim, requires_grad=True)
    with torch.enable_grad():
        zp = P(z, a, u)[0]
        J = torch.stack([torch.autograd.grad(zp[:, d].sum(), u, retain_graph=True)[0] for d in range(zp.shape[-1])], 1)   # n,D,K
    _, _, Vh = torch.linalg.svd(J)
    v = Vh[:, 0, :]
    v = v * torch.sign((J[:, 1, :] * v).sum(-1))[:, None]
    vbar = v.mean(0)
    return (vbar / vbar.norm()).detach()


@torch.no_grad()
def atom_weight(jepa, vbar, c=2.0, n=3000, seed=777, Mu=256):
    """Global P(+): fraction of u ~ N(0,I) whose y-displacement lies above the midpoint of the two atoms."""
    P = jepa.pred
    z, a = _random_states(n, seed)
    dp = P(z, a, c * vbar.expand(n, -1))[0][:, 1]
    dm = P(z, a, -c * vbar.expand(n, -1))[0][:, 1]
    mid = 0.5 * (dp + dm)
    g = torch.Generator().manual_seed(seed + 1)
    U = torch.randn(Mu, P.noise_dim, generator=g)
    w = []
    for i in range(0, n, 500):
        zi, ai = z[i:i + 500, None, :].expand(-1, Mu, -1), a[i:i + 500, None, :].expand(-1, Mu, -1)
        y = P(zi, ai, U[None].expand(zi.shape[0], -1, -1))[0][..., 1]
        w.append((y > mid[i:i + 500, None]).float().mean(1))
    return float(torch.cat(w).mean())


# ----------------------------------------------------------------------------------------------- sign design
_OA_CACHE = {}


def oa_design(M=8, H=10, seed=0, iters=60000):
    """+-1 matrix (M,H) with balanced columns and minimal pairwise column correlation (simulated-annealing frame-potential search)."""
    key = (M, H, seed)
    if key in _OA_CACHE:
        return _OA_CACHE[key]
    rng = np.random.default_rng(seed)
    def crit(X):
        G = X.T @ X
        off = G - np.diag(np.diag(G))
        return (off ** 2).sum() + 20.0 * (X.sum(0) ** 2).sum() + 20.0 * ((X.sum(1)) ** 2).sum() / H
    best, bc = None, 1e18
    for r in range(6):
        X = rng.choice([-1.0, 1.0], size=(M, H)); c = crit(X)
        for it in range(iters // 6):
            T = 30.0 * (1 - it / (iters // 6)) + 0.01
            i, j = rng.integers(M), rng.integers(H)
            X[i, j] *= -1; c2 = crit(X)
            if c2 <= c or rng.random() < math.exp(-(c2 - c) / T): c = c2
            else: X[i, j] *= -1
        if c < bc: best, bc = X.copy(), c
    _OA_CACHE[key] = best
    return best


class N2Noise:
    """Shared (CRN) node designs along vbar.  Same interface as d6_noise.NoiseSource (share=True)."""

    def __init__(self, kind, vbar, K=8, c_oa=1.5, U_learned=None, oa_seed=0):
        self.kind, self.vbar, self.K, self.c_oa = kind, vbar, K, c_oa
        self.U_learned = U_learned
        self.oa_seed = oa_seed
        self.fresh = True
        self._reset_mask = None

    def begin_replan(self, reset_mask=None):
        self.fresh = True

    def _q(self, B, M, H, gen, dev):
        k = self.kind
        if k == 'dir_iid':
            return torch.randn(B, M, H, device=dev, generator=gen)
        if k in ('dir_lhs', 'dir_lhs_anti'):
            Mb = M // 2 if k.endswith('anti') else M
            perm = _rand((B, H, Mb), gen, dev).argsort(-1).permute(0, 2, 1)          # (B,Mb,H) stratum index
            q = _ndtri((perm.float() + _rand((B, Mb, H), gen, dev)) / Mb)
            return torch.cat([q, -q], 1) if k.endswith('anti') else q
        if k == 'dir_oa':
            X = torch.tensor(oa_design(M, H, self.oa_seed), dtype=torch.float32, device=dev)          # (M,H)
            flip = (torch.randint(0, 2, (B, 1, H), device=dev, generator=gen) * 2 - 1).float()
            colperm = _rand((B, H), gen, dev).argsort(-1)                                              # (B,H)
            Xp = X[:, colperm.reshape(-1)].reshape(M, B, H).permute(1, 0, 2)                           # (B,M,H)
            return self.c_oa * Xp * flip
        raise ValueError(k)

    def get(self, E, N, M, H, gen, dev):
        """(E,1,M,H,K)"""
        if self.kind == 'learned':
            U = self.U_learned.to(dev)
            return U[None, None].expand(E, 1, M, H, self.K)
        q = self._q(E, M, H, gen, dev)                                                   # (E,M,H)
        xi = torch.randn(E, M, H, self.K, device=dev, generator=gen)
        xi = xi - (xi * self.vbar.to(dev)).sum(-1, keepdim=True) * self.vbar.to(dev)       # complement only
        return (q[..., None] * self.vbar.to(dev) + xi)[:, None]


class AtomLatentModel(LatentModel):
    """Deterministic atom-cloud belief propagation (see module docstring).  Rollout features are (E,N,L,D+3):
    [z, acc, failed, weight]; `cost` returns L * w * cost so that the usual mean over the last axis is the weighted mean."""

    def __init__(self, jepa, variant, vbar, K=5, p_plus=0.5, c=2.0, nodes=None, weights=None, **kw):
        super().__init__(jepa, variant, **kw)
        self.vbar, self.K = vbar, K
        self.nodes = [c, -c] if nodes is None else list(nodes)
        self.weights = [p_plus, 1 - p_plus] if weights is None else list(weights)
        self.max_atoms = 0

    @staticmethod
    def _merge_once(z, w, acc, f):
        E, N, L, D = z.shape
        dist = ((z[:, :, :, None, :] - z[:, :, None, :, :]) ** 2).sum(-1)
        ww = w[..., :, None] * w[..., None, :] / (w[..., :, None] + w[..., None, :] + 1e-12)
        cost = ww * dist
        eye = torch.eye(L, dtype=torch.bool, device=z.device)
        cost = cost.masked_fill(eye, float('inf')).reshape(E, N, L * L)
        k = cost.argmin(-1)
        i, j = k // L, k % L
        ar = torch.arange(L, device=z.device)
        oh_i = (ar == i[..., None]).float(); oh_j = (ar == j[..., None]).float()
        wi = (w * oh_i).sum(-1); wj = (w * oh_j).sum(-1); wn = wi + wj + 1e-12
        def mix(x):
            xi_ = (x * oh_i.unsqueeze(-1)).sum(2) if x.dim() == 4 else (x * oh_i).sum(-1)
            xj_ = (x * oh_j.unsqueeze(-1)).sum(2) if x.dim() == 4 else (x * oh_j).sum(-1)
            return xi_, xj_
        zi, zj = mix(z); ai, aj = mix(acc); fi, fj = mix(f)
        zm = (wi[..., None] * zi + wj[..., None] * zj) / wn[..., None]
        am = (wi * ai + wj * aj) / wn; fm = (wi * fi + wj * fj) / wn
        keep = ar[None, None, :L - 1]
        src = torch.where(keep < j[..., None], keep, keep + 1)                           # drop atom j
        def take(x):
            if x.dim() == 4:
                return torch.gather(x, 2, src[..., None].expand(E, N, L - 1, x.shape[-1]))
            return torch.gather(x, 2, src)
        z2 = z * (1 - oh_i[..., None]) + zm[:, :, None, :] * oh_i[..., None]
        w2 = w * (1 - oh_i) + wn[..., None] * oh_i
        a2 = acc * (1 - oh_i) + am[..., None] * oh_i
        f2 = f * (1 - oh_i) + fm[..., None] * oh_i
        return take(z2), take(w2), take(a2), take(f2)

    def _rollout(self, z0, acts, M, gen, goal=None):
        E, N, H, A = acts.shape
        D = z0.shape[-1]
        dev = z0.device
        vb = self.vbar.to(dev)
        z = z0[:, None, None, :].expand(E, N, 1, D).contiguous()
        w = torch.ones(E, N, 1, device=dev)
        acc = torch.zeros(E, N, 1, device=dev); fl = torch.zeros(E, N, 1, device=dev)
        for t in range(H):
            L = z.shape[2]
            B = len(self.nodes)
            a = acts[:, :, t, None, :].expand(E, N, B * L, A)
            u = torch.cat([nd * vb.expand(E, N, L, -1) for nd in self.nodes], 2)
            zc = torch.cat([z] * B, 2)
            zn = self.m.pred(zc, a, u)[0]
            w = torch.cat([w * wt for wt in self.weights], 2)
            acc = torch.cat([acc] * B, 2); fl = torch.cat([fl] * B, 2)
            if self.stage_w > 0:
                acc = acc + ((zn - goal[:, None, None, :]) ** 2).sum(-1) / H
            if self.anchor is not None:
                hit = (((zn - self.anchor.z) ** 2).sum(-1) < self.anchor.tau).float()
                fl = fl + (1 - fl) * hit
            z = zn
            while z.shape[2] > self.K:
                z, w, acc, fl = self._merge_once(z, w, acc, fl)
        self.max_atoms = z.shape[2]
        return torch.cat([z, acc[..., None], fl[..., None], w[..., None]], -1)

    def cost(self, feat, zg):
        w = feat[..., -1]
        c = super().cost(feat[..., :-1], zg)
        return c * w * feat.shape[2]


def make_n2_model(name, mem, variant, learned_path=None, **kw):
    """Returns a planning model for an n2 scheme name, or None if `name` is not an n2 scheme."""
    if name.startswith('atoms'):
        K = int(name[5:].split('_')[0]) if len(name) > 5 else 5
        vbar = active_direction(mem)
        p = atom_weight(mem, vbar) if name.endswith('_fit') else 0.5
        return AtomLatentModel(mem, variant, vbar, K=K, p_plus=p, **kw)
    if name.startswith('ut'):                                   # unscented / Gauss-Hermite-3 nodes along vbar (negative control)
        K = int(name[2:]) if len(name) > 2 else 3
        r3 = math.sqrt(3.0)
        return AtomLatentModel(mem, variant, active_direction(mem), K=K, nodes=[0.0, r3, -r3], weights=[2 / 3, 1 / 6, 1 / 6], **kw)
    if name.startswith('dir_') or name == 'learned':
        vbar = active_direction(mem)
        U = None
        if name == 'learned':
            d = torch.load(learned_path)
            xi = d['xi'] - (d['xi'] * vbar).sum(-1, keepdim=True) * vbar
            U = d['q'][..., None] * vbar + xi
        return CoupledLatentModel(mem, variant, noise=N2Noise(name, vbar, U_learned=U), **kw)
    return None
