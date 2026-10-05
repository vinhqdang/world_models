"""S1 Phase A helpers: CEM populations with recorded scores, and true-simulator costs for arbitrary plans (cliff_hi, fixed encoder)."""
import numpy as np, torch
from .stochnav import StochNav
from .jepa import JEPA, FailureAnchor, LatentModel, observe
from .d6_model import CoupledLatentModel
from .d6_noise import NoiseSource

VARIANT, H, MT = 'cliff_hi', 10, 200


def load_model(ckpt, scheme='indep'):
    ck = torch.load(ckpt, map_location='cpu'); a = ck['args']
    mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel'))
    mem.load_state_dict(ck['state']); mem.eval()
    anchor = FailureAnchor(mem.encode, VARIANT, mem.obs, device='cpu')
    e0 = StochNav(VARIANT, 1, 4242); sp, gp = e0.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, VARIANT, mem.obs))
        zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, VARIANT, mem.obs))
    scale = float(((zs - zg_) ** 2).sum(-1).median())
    kw = dict(stochastic=True, stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
    if scheme == 'indep':
        lm = LatentModel(mem, VARIANT, crn=False, **kw)
    elif scheme == 'crn':       # one noise set shared by all candidates AND all CEM iterations of a call (per state)
        lm = CoupledLatentModel(mem, VARIANT, noise=NoiseSource(gen_kind='iid', share=True, sticky='replan'), **kw)
    else:
        raise ValueError(scheme)
    return mem, anchor, lm


def make_states(n, seed, frac_start=0.25):
    """Mix of start states and mid-path states (on/near the start->goal line, y pushed towards the pit edge), outside the pit."""
    env = StochNav(VARIANT, 1, 0); env.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 10_000)
    s, g = env.sample_start_goal(n)
    is_start = rng.random(n) < frac_start
    t = np.where(is_start, 0.0, rng.uniform(0.05, 0.80, n))[:, None]
    P0 = s + t * (g - s)
    P0[:, 1] += np.where(is_start, 0.0, rng.uniform(-0.06, 0.10, n))
    P0 = np.clip(P0, 0.02, 0.98)
    bad = env._in(P0, env.pit); P0[bad, 1] = 0.31
    P0[:, 1] = np.where((~is_start) & (P0[:, 1] < 0.31) & (P0[:, 0] > 0.25) & (P0[:, 0] < 0.75), 0.31, P0[:, 1])
    return P0.astype(np.float32), g.astype(np.float32), is_start


@torch.no_grad()
def cem_pops(lm, z0, zg, N, M, iters, gen, ne_frac=0.1, init_std=0.7, keep=(0, -1)):
    """Same update as selwm.risk_plan.cem (mean risk, no shrinkage) but returns populations of iterations in `keep`
    with J (E,N), model failure fraction (E,N) and the plan tensors."""
    E = z0.shape[0]; A = 2
    mu = torch.zeros(E, 1, H, A); sd = torch.full((E, 1, H, A), init_std)
    ne = max(3, int(N * ne_frac)); out = {}
    if hasattr(lm, 'begin_replan'): lm.begin_replan()
    for it in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, generator=gen)).clamp(-1, 1)
        feat = lm.rollout(z0, acts, M, gen, zg)
        c = lm.cost(feat, zg)                                    # (E,N,M)
        s = c.mean(-1)
        if it in keep or (it - iters) in keep:
            out[it] = dict(acts=acts.clone(), J=s.clone(), pf=feat[..., -1].mean(-1), c=c)
        idx = s.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1, keepdim=True); sd = el.std(1, keepdim=True).clamp_min(0.05)
    return out


@torch.no_grad()
def true_cost(lm, mem, anchor, p0, zg, acts, Mt, gen):
    """Cost functional of lm on Mt TRUE-simulator particles (encoded by the frozen encoder). Returns (E,N) mean cost and (E,N) true pit-fall prob."""
    env = StochNav(VARIANT, 1, 0)
    pit = torch.tensor(env.pit, dtype=torch.float32)
    wind, step = env.wind_base, env.step_size
    E, N = acts.shape[:2]
    def in_pit(p):
        x, y = p[..., 0:1], p[..., 1:2]
        return ((x >= pit[:, 0]) & (x <= pit[:, 2]) & (y >= pit[:, 1]) & (y <= pit[:, 3])).any(-1)
    p = p0[:, None, None, :].expand(E, N, Mt, 2).clone()
    stuck = torch.zeros(E, N, Mt, dtype=torch.bool)
    acc = torch.zeros(E, N, Mt); failed = torch.zeros(E, N, Mt)
    for k in range(H):
        a_ = acts[:, :, k, None, :].clamp(-1, 1)
        pn = p + step * a_
        sign = (torch.randint(0, 2, (E, N, Mt), generator=gen) * 2 - 1).float()
        pn[..., 1] = pn[..., 1] + wind * sign
        pn = pn.clamp(0, 1)
        pn = torch.where(stuck[..., None], p, pn)
        stuck = stuck | in_pit(pn)
        p = pn
        z = mem.encode(observe(p, stuck, VARIANT, mem.obs))
        acc = acc + ((z - zg[:, None, None, :]) ** 2).sum(-1) / H
        failed = torch.maximum(failed, (((z - anchor.z) ** 2).sum(-1) < anchor.tau).float())
    feat = torch.cat([z, acc[..., None], failed[..., None]], -1)
    return lm.cost(feat, zg).mean(-1), stuck.float().mean(-1)
