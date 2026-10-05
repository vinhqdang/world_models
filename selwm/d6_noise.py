"""Coupled / variance-reduced noise sets for particle planning with the energy-score (es) latent model.

NoiseSource builds noise tensors u of shape (E, Nb, M, H, K) with Nb = 1 (shared across candidates, i.e. common
random numbers) or Nb = N (independent per candidate), from one of several generators:

  iid     plain Gaussian
  sobol   scrambled (random digital shift) Sobol points, K dims per step; particle order independently permuted per step
  sobol80 same but one H*K-dimensional Sobol point per particle (no permutation)
  halton  Cranley-Patterson rotated Halton points, K dims per step, permuted per step
  lhs     Latin hypercube: every marginal is stratified, independent permutations per dim

optionally antithetic (first M/2 particles drawn, second half is the negation), and with a stickiness policy:

  call    re-drawn at every model.rollout call (every CEM iteration)
  replan  drawn at begin_replan() and kept across the CEM iterations of that replan
  chain   as replan, but the set of the previous replan is shifted one step in time (the last step is fresh) and mixed
          with a fresh set with weight `refresh` (u = sqrt(1-r^2) u_old + r eps); refresh = 1 equals `replan`.
"""
import math
import torch
from scipy.stats import qmc

_SQRT2 = math.sqrt(2.0)
_BITS = 30
_cache = {}


def _sobol_int(d, m):
    key = ('sobol', d, m)
    if key not in _cache:
        s = qmc.Sobol(d, scramble=False).random_base2(m)               # (2^m, d) floats, dyadic
        _cache[key] = torch.tensor((s * (1 << _BITS)).astype('int64'))
    return _cache[key]


def _halton_pts(d, M):
    key = ('halton', d, M)
    if key not in _cache:
        _cache[key] = torch.tensor(qmc.Halton(d, scramble=False).random(M + 1)[1:], dtype=torch.float64)
    return _cache[key]


def _ndtri(u):
    u = u.clamp(1e-6, 1 - 1e-6)
    return _SQRT2 * torch.erfinv(2 * u - 1)


def _rand(shape, gen, dev, dtype=torch.float32):
    return torch.rand(shape, device=dev, generator=gen, dtype=dtype)


def _uniform_set(kind, B, Mb, H, K, gen, dev):
    """Uniform (0,1) point sets, shape (B, Mb, H, K)."""
    if kind == 'sobol':
        m = int(math.ceil(math.log2(Mb)))
        S = _sobol_int(K, m)[:Mb]                                       # (Mb,K) ints
        mask = torch.randint(0, 1 << _BITS, (B, 1, H, K), device=dev, generator=gen)
        x = torch.bitwise_xor(S[None, :, None, :].to(dev), mask).double() / (1 << _BITS)
        perm = _rand((B, H, Mb), gen, dev).argsort(-1)                  # (B,H,Mb)
        x = torch.gather(x, 1, perm.permute(0, 2, 1)[..., None].expand(B, Mb, H, K))
        return x.float()
    if kind == 'sobol80':
        m = int(math.ceil(math.log2(Mb)))
        S = _sobol_int(H * K, m)[:Mb]
        mask = torch.randint(0, 1 << _BITS, (B, 1, H * K), device=dev, generator=gen)
        x = torch.bitwise_xor(S[None].to(dev), mask).double() / (1 << _BITS)
        return x.reshape(B, Mb, H, K).float()
    if kind == 'halton':
        P = _halton_pts(K, Mb).to(dev)                                  # (Mb,K)
        shift = _rand((B, 1, H, K), gen, dev, torch.float64)
        x = (P[None, :, None, :] + shift) % 1.0
        perm = _rand((B, H, Mb), gen, dev).argsort(-1)
        x = torch.gather(x, 1, perm.permute(0, 2, 1)[..., None].expand(B, Mb, H, K))
        return x.float()
    if kind == 'lhs':
        perm = _rand((B, H, K, Mb), gen, dev).argsort(-1).permute(0, 3, 1, 2)       # (B,Mb,H,K) strata index
        return ((perm.float() + _rand((B, Mb, H, K), gen, dev)) / Mb)
    raise ValueError(kind)


class NoiseSource:
    def __init__(self, gen_kind='iid', anti=False, share=False, sticky='call', refresh=1.0, K=8):
        self.gen_kind, self.anti, self.share, self.sticky, self.refresh, self.K = gen_kind, anti, share, sticky, refresh, K
        self.state = None
        self.fresh_replan = True

    def name(self):
        s = self.gen_kind + ('+anti' if self.anti else '') + ('|crn' if self.share else '|ind')
        if self.sticky != 'call':
            s += '|' + self.sticky + (f'{self.refresh:g}' if self.sticky == 'chain' else '')
        return s

    # ---- draws
    def draw(self, B, M, H, gen, dev):
        """(B, M, H, K) standard normal noise."""
        K = self.K
        Mb = M // 2 if self.anti else M
        if self.gen_kind == 'iid':
            u = torch.randn(B, Mb, H, K, device=dev, generator=gen)
        else:
            u = _ndtri(_uniform_set(self.gen_kind, B, Mb, H, K, gen, dev))
        if self.anti:
            u = torch.cat([u, -u], 1)
        return u

    def begin_replan(self, reset_mask=None):
        self.fresh_replan = True
        self._reset_mask = reset_mask

    def get(self, E, N, M, H, gen, dev):
        """Noise tensor (E, Nb, M, H, K) for one rollout call."""
        Nb = 1 if self.share else N
        if self.sticky == 'call':
            return self.draw(E * Nb, M, H, gen, dev).reshape(E, Nb, M, H, self.K)
        if self.fresh_replan or self.state is None or self.state.shape != (E, Nb, M, H, self.K):
            new = self.draw(E * Nb, M, H, gen, dev).reshape(E, Nb, M, H, self.K)
            if self.sticky == 'chain' and self.state is not None and self.state.shape == new.shape and self.refresh < 1.0:
                old = torch.cat([self.state[:, :, :, 1:], new[:, :, :, -1:]], 3)           # shift in time
                r = self.refresh
                new_mixed = math.sqrt(1 - r * r) * old + r * new
                rm = getattr(self, '_reset_mask', None)
                if rm is not None:
                    rm = torch.as_tensor(rm, device=dev)[:, None, None, None, None]
                    new_mixed = torch.where(rm, new, new_mixed)
                new = new_mixed
            self.state = new
            self.fresh_replan = False
        return self.state
