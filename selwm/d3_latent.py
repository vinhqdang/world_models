"""Feedback rollouts for the learned stochastic latent model: each particle carries the sampled latent z_t and the
noise-free nominal latent zbar_t (same predictor with u = 0, one extra forward per step on a single particle per candidate);
the feedback acts on the (x, y) components of z_t - zbar_t."""
import torch
from .jepa import LatentModel


class LatentFB(LatentModel):
    def rollout_fb(self, z0, acts, K, M, gen, goal=None):
        with torch.no_grad():
            return self._rollout_fb(z0, acts, K, M, gen, goal)

    def _rollout_fb(self, z0, acts, K, M, gen, goal):
        E, N, H, A = acts.shape
        det = (not self.stochastic) or self.m.kind == 'det'
        Me = 1 if det else M
        D = z0.shape[-1]
        z = z0[:, None, None, :].expand(E, N, Me, D)
        zbar = z0[:, None, None, :].expand(E, N, 1, D)
        acc = torch.zeros(E, N, Me, device=z0.device)
        failed = torch.zeros(E, N, Me, device=z0.device)
        for t in range(H):
            an = acts[:, :, t, None, :].clamp(-1, 1)                          # (E,N,1,A)
            Kt = K[:, :, min(t, K.shape[2] - 1), None, :]                       # (E,N,1,A)
            dev = (z - zbar)[..., :A]
            a = (an + Kt * dev).clamp(-1, 1)                                    # (E,N,Me,A)
            if self.m.kind == 'es' and not det:
                u = torch.randn(E, N, Me, self.m.noise_dim, device=z0.device, generator=gen)
                z = self.spread_next(z, a, u)
                zbar = self.m.pred(zbar, an, torch.zeros(E, N, 1, self.m.noise_dim, device=z0.device))[0]
            elif det and self.m.kind != 'det':
                z = self.m.pred(z, a)[0]
                zbar = self.m.pred(zbar, an)[0]
            else:
                z = self.m.sample_next(z, a, gen=gen)
                zbar = self.m.pred(zbar, an)[0]
            if self.stage_w > 0:
                acc = acc + ((z - goal[:, None, None, :]) ** 2).sum(-1) / H
            if self.anchor is not None:
                failed = torch.maximum(failed, (((z - self.anchor.z) ** 2).sum(-1) < self.anchor.tau).float())
        return torch.cat([z, acc[..., None], failed[..., None]], -1)
