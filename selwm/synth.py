"""Synthetic planning problem with a known, planted optimism structure.

Plans a in R^D. True cost  J(a) = ||a - a*||^2 / D  (context a* differs per round).
Model cost  Jhat(a) = J(a) + eps(a),  eps(a) = (s0 + s1 ||a||^2 / D) * f(a),
f = random Fourier-feature field (unit variance, correlation length ell): model error is a smooth
random field over plan space that is larger away from the data support (the origin).
"""
import math
import torch


class SynthWorld:
    def __init__(self, D=20, M=256, ell=1.0, s0=0.02, s1=0.15, seed=0, a_star_std=0.4):
        g = torch.Generator().manual_seed(seed)
        self.D, self.M, self.s0, self.s1, self.a_star_std = D, M, s0, s1, a_star_std
        self.W = torch.randn(D, M, generator=g) / ell
        self.ph = torch.rand(M, generator=g) * 2 * math.pi
        self.c = torch.randn(M, generator=g)

    def new_contexts(self, B, gen):
        return torch.randn(B, self.D, generator=gen) * self.a_star_std

    def J(self, a, a_star):                     # a: (B,N,D), a_star: (B,D)
        return ((a - a_star[:, None]) ** 2).sum(-1) / self.D

    def field(self, a):                         # (B,N,D) -> (B,N)
        return (torch.cos(a @ self.W + self.ph) * self.c).sum(-1) / math.sqrt(self.M)

    def Jhat(self, a, a_star):
        amp = self.s0 + self.s1 * (a ** 2).sum(-1) / self.D
        return self.J(a, a_star) + amp * self.field(a)
