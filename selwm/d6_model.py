"""Drop-in LatentModel subclass whose es noise comes from a NoiseSource (coupled / variance-reduced noise)."""
import torch
from .jepa import LatentModel
from .d6_noise import NoiseSource


class CoupledLatentModel(LatentModel):
    def __init__(self, jepa, variant, noise=None, **kw):
        super().__init__(jepa, variant, **kw)
        self.noise = noise if noise is not None else NoiseSource()

    def begin_replan(self, reset_mask=None):
        self.noise.begin_replan(reset_mask)

    def _rollout(self, z0, acts, M, gen, goal=None):
        E, N, H, A = acts.shape
        det = (not self.stochastic) or self.m.kind == 'det'
        if det or self.m.kind != 'es':
            return super()._rollout(z0, acts, M, gen, goal)
        Me = M
        u_all = self.noise.get(E, N, Me, H, gen, z0.device)                     # (E,Nb,M,H,K)
        z = z0[:, None, None, :].expand(E, N, Me, z0.shape[-1])
        acc = torch.zeros(E, N, Me, device=z0.device)
        failed = torch.zeros(E, N, Me, device=z0.device)
        for t in range(H):
            a = acts[:, :, t, None, :].expand(E, N, Me, A)
            u = u_all[:, :, :, t].expand(E, N, Me, self.m.noise_dim)
            z = self.spread_next(z, a, u)
            if self.stage_w > 0:
                acc = acc + ((z - goal[:, None, None, :]) ** 2).sum(-1) / H
            if self.anchor is not None:
                failed = torch.maximum(failed, (((z - self.anchor.z) ** 2).sum(-1) < self.anchor.tau).float())
        return torch.cat([z, acc[..., None], failed[..., None]], -1)


SCHEMES = {
    # name: NoiseSource kwargs
    'indep':        dict(gen_kind='iid'),
    'crn':          dict(gen_kind='iid', share=True),
    'crn_replan':   dict(gen_kind='iid', share=True, sticky='replan'),
    'anti':         dict(gen_kind='iid', anti=True),
    'crn_anti':     dict(gen_kind='iid', share=True, anti=True),
    'sobol':        dict(gen_kind='sobol'),
    'crn_sobol':    dict(gen_kind='sobol', share=True),
    'crn_sobol80':  dict(gen_kind='sobol80', share=True),
    'crn_halton':   dict(gen_kind='halton', share=True),
    'crn_lhs':      dict(gen_kind='lhs', share=True),
    'crn_sobol_anti': dict(gen_kind='sobol', share=True, anti=True),
    'crn_lhs_anti': dict(gen_kind='lhs', share=True, anti=True),
    'crn_anti_replan': dict(gen_kind='iid', share=True, anti=True, sticky='replan'),
    'crn_sobol_anti_replan': dict(gen_kind='sobol', share=True, anti=True, sticky='replan'),
    'crn_sobol_anti_chain5': dict(gen_kind='sobol', share=True, anti=True, sticky='chain', refresh=0.5),
    'crn_anti_chain5': dict(gen_kind='iid', share=True, anti=True, sticky='chain', refresh=0.5),
    'crn_anti_chain3': dict(gen_kind='iid', share=True, anti=True, sticky='chain', refresh=0.3),
    'crn_lhs_anti_replan': dict(gen_kind='lhs', share=True, anti=True, sticky='replan'),
}


def make_noise(name):
    return NoiseSource(**SCHEMES[name])
