"""S2 Phase B: closed-loop selection-calibrated certified planning (SCC) on cliff_hi with learned energy-score models.

Contents
  load_models        encoder-frozen es model, failure anchor, chained-CRN planning model and an independent-noise model
  cem_pop            the D9 open-loop CEM (identical update) that also returns the final population
  pi0_action         the conservative fallback controller (fixed standoff from the pit)
  SlotEnv            vectorised copy of StochNav dynamics with per-(seed, slot, episode) start/goal AND wind-sign streams
  OutcomeTracker     delayed realised outcomes (fall within the next H steps of the executed trajectory)
  conformal helpers  split-conformal quantile, SCC rule
"""
import math
import numpy as np, torch
from .stochnav import StochNav
from .jepa import JEPA, FailureAnchor, LatentModel, observe
from .d6_model import CoupledLatentModel, make_noise
from .noise_aware import NoiseAwareScorer

VARIANT, H, A, KAPPA = 'cliff_hi', 10, 2, 3.0
EP_LEN = 120

# ---------------------------------------------------------------------------------------------- models
def load_models(ckpt, scheme='crn_chain3'):
    ck = torch.load(ckpt, map_location='cpu'); a = ck['args']
    mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel'))
    mem.load_state_dict(ck['state']); mem.eval()
    assert mem.kind == 'es'
    anchor = FailureAnchor(mem.encode, VARIANT, mem.obs, device='cpu')
    e0 = StochNav(VARIANT, 1, 4242); sp, gp = e0.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, VARIANT, mem.obs))
        zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, VARIANT, mem.obs))
    scale = float(((zs - zg_) ** 2).sum(-1).median())
    kw = dict(stochastic=True, stage_w=1.0, anchor=anchor, kappa=KAPPA, scale=scale)
    lm = CoupledLatentModel(mem, VARIANT, noise=make_noise(scheme), **kw)       # planning model (chained CRN across candidates and replans)
    lm_ind = LatentModel(mem, VARIANT, crn=False, **kw)                         # fresh independent noise (escalation, re-scoring)
    return mem, anchor, lm, lm_ind


def count_rows(mem):
    """Wrap mem.pred.forward so that the number of single-state predictor evaluations is counted (D9 convention)."""
    cnt = dict(calls=0, rows=0)
    orig = mem.pred.forward
    def counted(z, *r, **k):
        cnt['calls'] += 1; cnt['rows'] += int(z.numel() // z.shape[-1])
        return orig(z, *r, **k)
    mem.pred.forward = counted
    return cnt


# ---------------------------------------------------------------------------------------------- planner
@torch.no_grad()
def cem_pop(lm, z0, zg, N, M, iters, gen, score_fn, elite_frac=0.1, init_std=0.7):
    """Same update as selwm.risk_plan.cem (no warm start). Returns dict with the executed plan (mean of the final elites),
    the final population (acts (E,N,H,2), score s (E,N), per-particle cost c (E,N,M), model failure fraction pf (E,N)) and elite order."""
    E = z0.shape[0]
    mu = torch.zeros(E, 1, H, A); sd = torch.full((E, 1, H, A), init_std)
    ne = max(3, int(N * elite_frac))
    for _ in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, generator=gen)).clamp(-1, 1)
        feat = lm.rollout(z0, acts, M, gen, zg)
        c = lm.cost(feat, zg)
        s = score_fn(c)
        idx = s.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu = el.mean(1, keepdim=True); sd = el.std(1, keepdim=True).clamp_min(0.05)
    return dict(plan=mu[:, 0], acts=acts, s=s, c=c, pf=feat[..., -1].mean(-1), order=idx)


@torch.no_grad()
def score_exec(lm, z0, zg, plan, M, gen):
    """Planner score of the executed plan with the planner's own (chained, shared) noise set: one extra candidate row set per env."""
    feat = lm.rollout(z0, plan[:, None], M, gen, zg)
    c = lm.cost(feat, zg)
    return c.mean(-1)[:, 0], feat[..., -1].mean(-1)[:, 0]          # total score j, model fall probability pf


def candidates(pop, K=4):
    """K=4 candidates: the executed (elite-mean) plan and the top K-1 population members by planner score."""
    top = pop['order'][:, :K - 1]
    mem_ = torch.gather(pop['acts'], 1, top[:, :, None, None].expand(-1, -1, H, A))
    return torch.cat([pop['plan'][:, None], mem_], 1)


@torch.no_grad()
def rescore(lm_ind, z0, zg, cand, Mp, gen):
    """Fresh independent-noise re-scoring of candidates with Mp particles: total score (E,K) and model fall probability (E,K)."""
    out_j, out_p = [], []
    for lo in range(0, z0.shape[0], 4):                              # chunk to bound memory
        sl = slice(lo, lo + 4)
        feat = lm_ind.rollout(z0[sl], cand[sl], Mp, gen, zg[sl])
        out_j.append(lm_ind.cost(feat, zg[sl]).mean(-1)); out_p.append(feat[..., -1].mean(-1))
    return torch.cat(out_j), torch.cat(out_p)


# ---------------------------------------------------------------------------------------------- fallback controller
PI0 = dict(y_so=0.85, hold=0.05, v0=0.5, x_clear=0.80, x_in=0.14)       # fixed before any evaluation, see results/s2/pi0_check.txt
PI0_FAST = dict(PI0, v0=1.0)                                              # reference only


def pi0_action(p, g, cfg=PI0):
    """Conservative goal-directed controller with a fixed standoff from the pit (pit: x in [0.25,0.75], y <= 0.30).
    While x < x_clear it steers y to the standoff line y_so (= pit top + 0.55) with gain 1/0.05 and advances in x at the fraction v0 of the
    maximal step; inside the pit's x-range (+margin) it waits in x if y < y_so - hold. For x >= x_clear (past the pit) it
    steers to the goal with a proportional law. Uses the observed position only (no model)."""
    x, y = p[:, 0], p[:, 1]
    ay = np.clip((cfg['y_so'] - y) / 0.05, -1, 1)
    zone = (x > cfg['x_in']) & (x < cfg['x_clear'])
    ax = np.where(zone & (y < cfg['y_so'] - cfg['hold']), 0.0, cfg['v0'])
    a_goal = np.clip((g - p) / 0.05, -1, 1)
    a = np.stack([ax, ay], 1)
    return np.where((x >= cfg['x_clear'])[:, None], a_goal, a)


# ---------------------------------------------------------------------------------------------- environment
class SlotEnv:
    """E parallel slots. The j-th episode of slot i has start/goal from default_rng([8000+off, seed, i, j']) and wind-sign sequence
    from default_rng([9000+off, seed, i, j]); both are identical across arms, so arms see the same start/goal and the same wind
    sequence by episode step (planner randomness and actions differ). Dynamics = StochNav.transition with a given sign.
    `phase` (int) separates the generator streams of burn-in and evaluation phases."""

    def __init__(self, seed, E=16, wind=0.09, phase=0):
        self.seed, self.E, self.phase = seed, E, phase
        self.ref = StochNav(VARIANT, 1, 0)
        self.wind, self.step_size, self.goal_r = wind, self.ref.step_size, self.ref.goal_r
        self.pit = self.ref.pit
        self.p = np.zeros((E, 2)); self.goal = np.zeros((E, 2))
        self.fell = np.zeros(E, bool); self.reached = np.zeros(E, bool)
        self.j = np.zeros(E, int); self.t_ep = np.zeros(E, int)
        self.signs = [None] * E
        for i in range(E): self._new_episode(i)

    def _new_episode(self, i):
        e = StochNav(VARIANT, 1, 0)
        e.rng = np.random.default_rng([8000 + 100 * self.phase, self.seed, i, int(self.j[i])])
        s, g = e.sample_start_goal(1)
        self.p[i], self.goal[i] = s[0], g[0]
        self.fell[i] = False; self.reached[i] = False; self.t_ep[i] = 0
        r = np.random.default_rng([9000 + 100 * self.phase, self.seed, i, int(self.j[i])])
        self.signs[i] = r.choice([-1.0, 1.0], size=EP_LEN + 5)

    def step(self, a, idx):
        """a: (len(idx),2) actions for slots idx."""
        a = np.clip(a, -1, 1)
        pn = self.p[idx] + self.step_size * a
        sg = np.array([self.signs[i][self.t_ep[i]] for i in idx])
        pn[:, 1] += self.wind * sg
        pn = np.clip(pn, 0.0, 1.0)
        fell = StochNav._in(pn, self.pit)
        self.p[idx] = pn
        self.fell[idx] = fell
        d = np.linalg.norm(pn - self.goal[idx], axis=1)
        self.reached[idx] = (~fell) & (d < self.goal_r)
        self.t_ep[idx] += 1

    def finished(self, idx):
        return self.fell[idx] | self.reached[idx] | (self.t_ep[idx] >= EP_LEN)

    def outcome(self, i):
        return 'fall' if self.fell[i] else ('success' if self.reached[i] else 'timeout')

    def next_episode(self, i):
        self.j[i] += 1; self._new_episode(i)


# ---------------------------------------------------------------------------------------------- delayed realised outcomes
class OutcomeTracker:
    """Registers per-step records (any payload) of env i and resolves them with the realised closed-loop outcome
    F = 1[the agent falls within the next H steps of the executed trajectory]. A record is resolved at age H (F=0 if no fall),
    immediately at a fall (F=1 for every pending record of the episode) and at episode end without fall (F=0)."""

    def __init__(self, E):
        self.pend = [[] for _ in range(E)]

    def add(self, i, payload):
        self.pend[i].append([0, payload])

    def after_step(self, i, fell, ended):
        """call after env step for slot i; returns list of (payload, F)."""
        out = []
        for rec in self.pend[i]: rec[0] += 1
        if fell:
            out = [(p, 1) for _, p in self.pend[i]]; self.pend[i] = []
        elif ended:
            out = [(p, 0) for _, p in self.pend[i]]; self.pend[i] = []
        else:
            keep = []
            for age, p in self.pend[i]:
                if age >= H: out.append((p, 0))
                else: keep.append([age, p])
            self.pend[i] = keep
        return out


# ---------------------------------------------------------------------------------------------- conformal helpers
def conf_quantile(R, alpha):
    """split-conformal: the ceil((1-alpha)(n+1))-th smallest residual (inf if it does not exist)."""
    R = np.sort(np.asarray(R)); n = len(R)
    k = int(math.ceil((1 - alpha) * (n + 1)))
    return float(R[k - 1]) if k <= n else float('inf')


def level_quantile(R_sorted, level):
    """empirical quantile of a sorted residual array at `level` in the conformal sense (level>=1 -> +inf, level<=0 -> min)."""
    n = len(R_sorted)
    k = int(math.ceil(level * (n + 1)))
    if k > n: return float('inf')
    return float(R_sorted[max(k, 1) - 1])


# ---------------------------------------------------------------------------------------------- true simulator (diagnostic / calibration)
@torch.no_grad()
def true_cost_w(lm, mem, anchor, p0, zg, acts, Mt, gen, wind):
    """Copy of selwm.s1_pop.true_cost with an explicit wind amplitude (identical output for wind=0.09).
    Returns mean planner-cost (E,N) of open-loop execution of `acts` on Mt true-simulator particles and the true pit-fall probability (E,N)."""
    env = StochNav(VARIANT, 1, 0)
    pit = torch.tensor(env.pit, dtype=torch.float32)
    step = env.step_size
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


# ---------------------------------------------------------------------------------------------- counterfactual replay of a committed plan
class ReplayLog:
    """Observed transitions of the current episode of every slot, used to replay a committed open-loop plan on the REALISED wind.
    The wind of an executed step is read off the observed transition (y' - y - 0.05 a_y, known kinematics, additive y-wind); steps whose y was
    clipped by the arena border, and steps after the episode ended, are unknown and get a random sign with the amplitude of the median
    observed |wind| of the window (0.09 if none). The replayed outcome F_replay in {0,1} is one sample of 'the plan, executed open loop for H
    steps, enters the pit', an unbiased sample of the true open-loop fall probability pT because the wind is independent of the actions."""

    def __init__(self, E, seed):
        self.h = [[] for _ in range(E)]
        self.rng = np.random.default_rng([4242, seed])
        self.pit = StochNav(VARIANT, 1, 0).pit

    def reset(self, i): self.h[i] = []

    def log(self, i, y_before, y_after, a_exec_y):
        self.h[i].append((float(y_before), float(y_after), float(np.clip(a_exec_y, -1, 1))))

    def replay(self, i, plan, p0, t0, step=0.05):
        hh = self.h[i][t0:t0 + H]
        known = [abs((ya - yb) - step * ay) for (yb, ya, ay) in hh if 1e-9 < ya < 1 - 1e-9]
        amp = float(np.median(known)) if known else 0.09
        w = []
        for (yb, ya, ay) in hh:
            if 1e-9 < ya < 1 - 1e-9: w.append((ya - yb) - step * ay)              # exact
            elif ya >= 1 - 1e-9 and yb + step * ay - amp < 1.0: w.append(+amp)       # clipped at the top: the wind was upwards
            elif ya <= 1e-9 and yb + step * ay + amp > 0.0: w.append(-amp)           # clipped at the bottom: the wind was downwards
            else: w.append(None)
        p = np.array(p0, float)
        for k in range(H):
            wk = w[k] if (k < len(w) and w[k] is not None) else amp * self.rng.choice([-1.0, 1.0])
            pn = p + step * np.clip(plan[k], -1, 1)
            pn[1] += wk
            pn = np.clip(pn, 0.0, 1.0)
            if bool(StochNav._in(pn[None], self.pit)[0]): return 1
            p = pn
        return 0
