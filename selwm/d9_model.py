"""Feedback rollouts (nominal path + deviation feedback) whose sampled path is driven by a NoiseSource shared across
candidates (common random numbers).  The nominal path always uses u = 0, so the deviation z_t - zbar_t is the response of
the sampled path to the (shared) noise only.  Open-loop rollouts (plain CEM) are inherited from CoupledLatentModel."""
import torch
from .d6_model import CoupledLatentModel, make_noise, SCHEMES  # noqa: F401


class CoupledLatentFB(CoupledLatentModel):
    def rollout_fb(self, z0, acts, K, M, gen, goal=None):
        with torch.no_grad():
            return self._rollout_fb(z0, acts, K, M, gen, goal)

    def _rollout_fb(self, z0, acts, K, M, gen, goal):
        E, N, H, A = acts.shape
        assert self.m.kind == 'es' and self.stochastic, 'd9 feedback rollouts need the stochastic es model'
        Me = M
        D = z0.shape[-1]
        u_all = self.noise.get(E, N, Me, H, gen, z0.device)                       # (E,Nb,M,H,Kn), Nb in {1,N}
        z = z0[:, None, None, :].expand(E, N, Me, D)
        zbar = z0[:, None, None, :].expand(E, N, 1, D)
        zero_u = torch.zeros(E, N, 1, self.m.noise_dim, device=z0.device)
        acc = torch.zeros(E, N, Me, device=z0.device)
        failed = torch.zeros(E, N, Me, device=z0.device)
        for t in range(H):
            an = acts[:, :, t, None, :].clamp(-1, 1)                              # (E,N,1,A)
            Kt = K[:, :, min(t, K.shape[2] - 1), None, :]                         # (E,N,1,A)
            a = (an + Kt * (z - zbar)[..., :A]).clamp(-1, 1)                      # (E,N,Me,A)
            u = u_all[:, :, :, t].expand(E, N, Me, self.m.noise_dim)
            z = self.spread_next(z, a, u)
            zbar = self.m.pred(zbar, an, zero_u)[0]
            if self.stage_w > 0:
                acc = acc + ((z - goal[:, None, None, :]) ** 2).sum(-1) / H
            if self.anchor is not None:
                failed = torch.maximum(failed, (((z - self.anchor.z) ** 2).sum(-1) < self.anchor.tau).float())
        return torch.cat([z, acc[..., None], failed[..., None]], -1)
