"""Small decoder-free JEPA world models: deterministic, Gaussian-head and energy-score (stochastic) predictors.

All variants share the encoder and the SIGReg regulariser of LeWorldModel (Maes et al. 2026): the loss is
    prediction loss + lambda * SIGReg(embeddings)
with no EMA target network and no decoder. They differ only in the predictor and the prediction loss:

  det   z' = z + f(z, a)                          loss: MSE(z', z_target)
  gauss z' ~ N(mu(z,a), diag sigma(z,a)^2)        loss: Gaussian NLL (heteroscedastic)
  es    z'_m = z + f(z, a, u_m), u_m ~ N(0, I)    loss: energy score of M samples against the target embedding
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regulariser (Epps-Pulley on random projections), as in LeJEPA / LeWM."""

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', w * window)

    def forward(self, z):                                  # z: (B, D)
        A = torch.randn(z.size(-1), self.num_proj, device=z.device)
        A = A / A.norm(dim=0)
        x_t = (z @ A).unsqueeze(-1) * self.t               # (B, P, K)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return (err @ self.weights).mean() * z.size(0)


class Encoder(nn.Module):
    def __init__(self, dim=64, size=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.GELU(),
        )
        n = 128 * (size // 16) ** 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):                                   # x: (B,3,S,S)
        return self.proj(self.fc(self.conv(x)))


class Predictor(nn.Module):
    def __init__(self, dim=64, adim=2, kind='det', noise_dim=8, hid=512, depth=3):
        super().__init__()
        self.kind, self.dim, self.noise_dim = kind, dim, noise_dim
        din = dim + adim + (noise_dim if kind == 'es' else 0)
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.GELU()]
            d = hid
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d, dim * (2 if kind == 'gauss' else 1))

    def forward(self, z, a, u=None):
        """z:(...,D) a:(...,A) u:(...,K) -> next embedding (mean for det/gauss; one sample for es) and log-sigma for gauss."""
        h = [z, a] + ([u] if self.kind == 'es' else [])
        o = self.out(self.net(torch.cat(h, -1)))
        if self.kind == 'gauss':
            mu, ls = o.chunk(2, -1)
            return z + mu, ls.clamp(-6, 2)
        return z + o, None


class JEPA(nn.Module):
    def __init__(self, kind='det', dim=64, noise_dim=8, sigreg_weight=0.09, size=48):
        super().__init__()
        self.kind = kind
        self.enc = Encoder(dim, size)
        self.pred = Predictor(dim, 2, kind, noise_dim)
        self.sigreg = SIGReg()
        self.lam = sigreg_weight
        self.noise_dim = noise_dim

    # ------------------------------------------------------------------ training
    def loss(self, x0, a, x1, M=8):
        z = self.enc(torch.cat([x0, x1], 0))
        z0, z1 = z.chunk(2, 0)
        reg = self.sigreg(z)
        if self.kind == 'det':
            zp, _ = self.pred(z0, a)
            pl = (zp - z1).pow(2).mean()
        elif self.kind == 'gauss':
            mu, ls = self.pred(z0, a)
            pl = (0.5 * ((z1 - mu) / ls.exp()).pow(2) + ls).mean()
        else:                                                         # energy score
            u = torch.randn(M, z0.shape[0], self.noise_dim, device=z0.device)
            zs, _ = self.pred(z0.expand(M, -1, -1), a.expand(M, -1, -1), u)           # (M,B,D)
            t1 = (zs - z1).norm(dim=-1).mean(0)                                        # E||X - y||
            d = (zs.unsqueeze(0) - zs.unsqueeze(1)).norm(dim=-1)                      # (M,M,B)
            t2 = d.sum((0, 1)) / (M * (M - 1))                                         # E||X - X'||
            pl = (t1 - 0.5 * t2).mean()
        return pl + self.lam * reg, dict(pred=float(pl.detach()), sigreg=float(reg.detach()))

    # ------------------------------------------------------------------ planning interface
    @torch.no_grad()
    def encode(self, x):
        return self.enc(x)

    @torch.no_grad()
    def sample_next(self, z, a, gen=None, u=None):
        """One predictive sample (or the mean for det/gauss-mean)."""
        if self.kind == 'es':
            if u is None:
                u = torch.randn(*z.shape[:-1], self.noise_dim, device=z.device, generator=gen)
            return self.pred(z, a, u)[0]
        if self.kind == 'gauss':
            mu, ls = self.pred(z, a)
            return mu + ls.exp() * torch.randn(mu.shape, device=z.device, generator=gen)
        return self.pred(z, a)[0]


class LatentModel:
    """Adapter used by selwm.risk_plan: rollouts in latent space, cost = squared distance to the goal embedding."""

    def __init__(self, jepa, variant, stochastic=True, crn=False, stage_w=0.0):
        self.m, self.variant, self.stochastic, self.crn, self.stage_w = jepa, variant, stochastic, crn, stage_w
        from .stochnav import render
        self.render = render

    def obs_to_latent(self, p, fell=None):
        x = self.render(p, self.variant, fell)
        return self.m.encode(x)

    def rollout(self, z0, acts, M, gen, goal=None):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=z0.is_cuda):
            return self._rollout(z0, acts, M, gen, goal).float()

    def _rollout(self, z0, acts, M, gen, goal=None):
        E, N, H, A = acts.shape
        det = (not self.stochastic) or self.m.kind == 'det'
        Me = 1 if det else M
        z = z0[:, None, None, :].expand(E, N, Me, z0.shape[-1])
        u_shared = None
        acc = torch.zeros(E, N, Me, device=z0.device)
        for t in range(H):
            a = acts[:, :, t, None, :].expand(E, N, Me, A)
            if self.m.kind == 'es' and not det:
                if self.crn:
                    if u_shared is None:
                        u_shared = torch.randn(E, 1, Me, H, self.m.noise_dim, device=z0.device, generator=gen)
                    u = u_shared[:, :, :, t].expand(E, N, Me, self.m.noise_dim)
                else:
                    u = torch.randn(E, N, Me, self.m.noise_dim, device=z0.device, generator=gen)
                z = self.m.sample_next(z, a, u=u)
            elif det and self.m.kind != 'det':
                z = self.m.pred(z, a)[0]                               # mean prediction of a gauss head
            else:
                z = self.m.sample_next(z, a, gen=gen)
            if self.stage_w > 0:
                acc = acc + ((z - goal[:, None, None, :]) ** 2).sum(-1) / H
        return torch.cat([z, acc[..., None]], -1) if self.stage_w > 0 else z

    def cost(self, feat, zg):
        if self.stage_w > 0:
            return ((feat[..., :-1] - zg[:, None, None, :]) ** 2).sum(-1) + self.stage_w * feat[..., -1]
        return ((feat - zg[:, None, None, :]) ** 2).sum(-1)


class EnsembleLatent:
    """PETS-style baseline: an ensemble of independently trained deterministic JEPAs.

    Each particle is rolled out by one member (round-robin); randomness comes only from member disagreement
    (epistemic), not from aleatoric noise. Latents of the members live in different spaces, so the latent state is the
    concatenation of the member latents and the per-particle cost is computed in the member's own space.
    """

    def __init__(self, members, variant, stage_w=0.0):
        self.members, self.variant, self.stage_w = members, variant, stage_w
        self.K = len(members)
        self.D = members[0].enc.proj[-1].out_features
        from .stochnav import render
        self.render = render

    def obs_to_latent(self, p, fell=None):
        x = self.render(p, self.variant, fell)
        with torch.no_grad():
            return torch.cat([m.encode(x) for m in self.members], -1)

    @torch.no_grad()
    def rollout(self, z0, acts, M, gen, goal=None):
        E, N, H, A = acts.shape
        out = []
        for m_idx in range(M):
            k = m_idx % self.K
            mem = self.members[k]
            zk = z0[:, None, k * self.D:(k + 1) * self.D].expand(E, N, self.D)
            gk = goal[:, None, k * self.D:(k + 1) * self.D]
            acc = torch.zeros(E, N, device=z0.device)
            z = zk
            for t in range(H):
                z = mem.pred(z, acts[:, :, t])[0]
                if self.stage_w > 0:
                    acc = acc + ((z - gk) ** 2).sum(-1) / H
            c = ((z - gk) ** 2).sum(-1) + self.stage_w * acc
            out.append(c)
        return torch.stack(out, 2)[..., None]                      # (E,N,M,1) per-particle costs

    def cost(self, feat, zg):
        return feat[..., 0]
