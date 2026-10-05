"""Shared helpers for the N1 (cross-fitted CEM) experiments."""
import sys
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.d6_noise import NoiseSource
from selwm.d6_model import SCHEMES
from selwm.n1_cf import CFLatent, CFOracle, cf_cem, equal_mb

# arm -> (noise scheme, decide, MA per iteration, MB, n_short).  Totals are checked to equal 3*32*8 = 768 rollouts.
ARMS = {
    'ol_indep':   ('indep',      'mean', [8, 8, 8], 0, 3),
    'ol_crn3':    ('crn_chain3', 'mean', [8, 8, 8], 0, 3),
    'cf_crn3':    ('crn_chain3', 'cf',   [7, 7, 7], 24, 3),         # S = {mean, 3 elites}: 672 + 4*24 = 768
    'cf_indep':   ('indep',      'cf',   [7, 7, 7], 24, 3),
    'cf_crn3_b':  ('crn_chain3', 'cf',   [6, 6, 6], 48, 3),         # 576 + 4*48
    'cf_crn3_c':  ('crn_chain3', 'cf',   [4, 4, 4], 96, 3),         # 384 + 4*96
    'cf_crn3_d':  ('crn_chain3', 'cf',   [8, 8, 4], 32, 3),         # 512+128 + 4*32
    'cf_indep_b': ('indep',      'cf',   [6, 6, 6], 48, 3),
    'cf_indep_c': ('indep',      'cf',   [4, 4, 4], 96, 3),
    'cf_indep_d': ('indep',      'cf',   [8, 8, 4], 32, 3),
    'ref_indep64': ('indep',     'mean', [64, 64, 64], 0, 3),      # 8x compute reference (headroom), not budget-matched
    'ref_crn64':  ('crn_chain3', 'mean', [64, 64, 64], 0, 3),
}


def check_budget(arm, iters=3, N=32, M=8):
    if arm.startswith('ref'):
        return
    sch, dec, MA, MB, ns = ARMS[arm]
    S = ns + 1
    tot = N * sum(MA) + (S * MB if dec == 'cf' else 0)
    assert tot == iters * N * M, (arm, tot)


def load_jepa(path):
    ck = torch.load(path, map_location='cpu'); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel'))
    m.load_state_dict(ck['state']); m.eval()
    return m


def start_goal_scale(mem, variant):
    e = StochNav(variant, 1, 4242)
    sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs))
        zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
    return float(((zs - zg_) ** 2).sum(-1).median())


class Planner:
    """Wraps a model (learned latent or oracle) + arm; call .plan(p, fell, goal, gen, diag) with numpy/torch positions."""

    def __init__(self, arm, model_kind, ckpt=None, variant='cliff_hi', H=10, N=32, iters=3, overrides=None):
        sch, dec, MA, MB, ns = ARMS[arm]
        if overrides:
            sch = overrides.get('scheme', sch); dec = overrides.get('decide', dec); MA = overrides.get('MA', MA)
            MB = overrides.get('MB', MB); ns = overrides.get('n_short', ns)
        self.rho = (overrides or {}).get('rho', 0.0)
        self.arm, self.dec, self.MA, self.MB, self.ns, self.H, self.N, self.iters = arm, dec, MA, MB, ns, H, N, iters
        self.model_kind = model_kind
        K = 8 if model_kind == 'learned' else 1
        self.noise = NoiseSource(**SCHEMES[sch], K=K)
        self.K = K
        if model_kind == 'learned':
            mem = load_jepa(ckpt)
            anchor = FailureAnchor(mem.encode, variant, mem.obs)
            self.lm = CFLatent(mem, variant, noise=self.noise, stochastic=True, stage_w=1.0, anchor=anchor, kappa=3.0,
                               scale=start_goal_scale(mem, variant))
            self.K = mem.noise_dim
            self.cnt = dict(rows=0)
            orig = mem.pred.forward
            cnt = self.cnt
            def counted(z, *r, **k):
                cnt['rows'] += int(z.numel() // z.shape[-1]); return orig(z, *r, **k)
            mem.pred.forward = counted
        else:
            self.lm = CFOracle(variant, stage_w=1.0)
            self.lm.noise = self.noise
            self.cnt = dict(rows=0)
        self.lm_noise_dim = self.K

    def latent(self, p, fell, goal):
        if self.model_kind == 'learned':
            return self.lm.obs_to_latent(p, fell), self.lm.obs_to_latent(goal)
        return p, goal

    def begin_replan(self, reset_mask=None):
        self.noise.begin_replan(reset_mask)

    def plan(self, p, fell, goal, gen, diag=False):
        z0, zg = self.latent(p, fell, goal)
        out = cf_cem(self.lm, z0, zg, self.H, self.N, self.iters, self.MA, self.MB, self.noise, gen, self.K,
                     n_short=self.ns, decide=self.dec, return_diag=diag, rho=self.rho)
        return out
