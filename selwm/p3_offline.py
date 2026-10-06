"""P3: offline (environment-free) validation statistics for learned latent world models used by a CEM planner.

Everything here uses only (a) a logged dataset of (position, action, next position, fall flag) produced by a behaviour
policy and (b) the candidate model itself.  The simulator is never stepped by a statistic; ground truth comes from a
separate closed-loop evaluation (experiments/d8_eval.py).

All suite models use the identity ('fixed') encoder, so latents are comparable across models.
"""
import math
import numpy as np
import torch
from .stochnav import StochNav, collect
from .jepa import JEPA, FailureAnchor, observe, LatentModel
from .noise_aware import NoiseAwareScorer
from .risk_plan import cem

VARIANT = 'cliff_hi'
H = 10
KAPPA, STAGE_W = 3.0, 1.0           # planner settings used by experiments/d8_eval.py


# ------------------------------------------------------------------------------------------ models
def load_model(path):
    ck = torch.load(path, map_location='cpu')
    a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel'))
    m.load_state_dict(ck['state']); m.eval()
    anchor = FailureAnchor(m.encode, VARIANT, m.obs)
    e = StochNav(VARIANT, 1, 4242)
    sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = m.encode(observe(torch.tensor(sp, dtype=torch.float32), None, VARIANT, m.obs))
        zg = m.encode(observe(torch.tensor(gp, dtype=torch.float32), None, VARIANT, m.obs))
    scale = float(((zs - zg) ** 2).sum(-1).median())
    lm = LatentModel(m, VARIANT, stochastic=True, crn=False, stage_w=STAGE_W, anchor=anchor, kappa=KAPPA, scale=scale)
    return m, lm, a


def enc(m, p, fell=None):
    with torch.no_grad():
        return m.encode(observe(p, fell, VARIANT, m.obs))


# ------------------------------------------------------------------------------------------ logged data
def collect_ordered(seed, n_envs=200, T=400, mix_goal=0.4):
    """Same behaviour policy as selwm.stochnav.collect but keeps the time order. Returns dict of (T, n_envs, .) arrays and `reset`."""
    env = StochNav(VARIANT, n_envs, seed)
    rng = np.random.default_rng(seed + 1)
    P = np.zeros((T, n_envs, 2), np.float32); A = np.zeros_like(P); P2 = np.zeros_like(P)
    F = np.zeros((T, n_envs), bool); F2 = np.zeros((T, n_envs), bool); R = np.zeros((T, n_envs), bool)
    a = rng.uniform(-1, 1, (n_envs, 2))
    goaldir = rng.random(n_envs) < mix_goal
    for t in range(T):
        a = 0.85 * a + 0.55 * rng.normal(size=(n_envs, 2))
        tgt = env.goal - env.p
        tgt = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-8)
        act = np.clip(np.where(goaldir[:, None], 0.7 * tgt + 0.6 * rng.normal(size=(n_envs, 2)), a), -1, 1)
        P[t] = env.p; F[t] = env.fell
        alive = ~(env.fell | env.reached)
        pn, fell = env.transition(env.p, act, rng)
        A[t] = act
        P2[t] = np.where(alive[:, None], pn, env.p)
        env.p = P2[t].astype(np.float64)
        env.fell |= alive & fell
        F2[t] = env.fell
        d = np.linalg.norm(env.p - env.goal, axis=1)
        env.reached |= alive & (~env.fell) & (d < env.goal_r)
        done = env.fell | env.reached | (rng.random(n_envs) < 0.01)
        R[t] = done
        if done.any():
            s, g = env.sample_start_goal(int(done.sum()))
            s = np.where(rng.random((len(s), 1)) < 0.6, rng.uniform(0.02, 0.98, (len(s), 2)), s)
            env.p[done] = s; env.goal[done] = g
            env.fell[done] = False; env.reached[done] = False
            goaldir[done] = rng.random(int(done.sum())) < mix_goal
    return dict(P=P, A=A, P2=P2, F=F, F2=F2, reset=R)


def make_windows(d, Hh=H):
    """All length-H windows that do not cross a reset and start alive.  Returns arrays (W,...)."""
    T, E = d['P'].shape[:2]
    idx_t, idx_e = [], []
    ok_run = np.ones((T - Hh + 1, E), bool)
    for k in range(Hh - 1):                                     # reset after step t+k breaks the window
        ok_run &= ~d['reset'][k:T - Hh + 1 + k]
    ok = ok_run & ~d['F'][:T - Hh + 1]
    ts, es = np.where(ok)
    P0 = d['P'][ts, es]
    acts = np.stack([d['A'][ts + k, es] for k in range(Hh)], 1)
    traj = np.stack([d['P2'][ts + k, es] for k in range(Hh)], 1)         # positions after each step
    fl = np.stack([d['F2'][ts + k, es] for k in range(Hh)], 1)
    return dict(p0=P0, acts=acts, traj=traj, fell=fl)


# ------------------------------------------------------------------------------------------ one-step predictive stats
@torch.no_grad()
def predict_samples(m, z, a, M=32, gen=None):
    """z (B,D), a (B,A) -> samples (M,B,D)."""
    zz = z[None].expand(M, *z.shape); aa = a[None].expand(M, *a.shape)
    if m.kind == 'det':
        return m.pred(zz, aa)[0]
    return m.sample_next(zz, aa, gen=gen)


def energy_score(samples, y):
    M = samples.shape[0]
    t1 = (samples - y[None]).norm(dim=-1).mean(0)
    d = (samples[None] - samples[:, None]).norm(dim=-1)
    t2 = d.sum((0, 1)) / (M * (M - 1))
    return t1 - 0.5 * t2


def onestep_stats(m, lm, V, M=32, seed=0, chunk=4000):
    """V: dict P,A,P2,F,F2 numpy. Returns per-sample arrays: ES, SE of mean, NLL (diag Gaussian), hazard prob/indicator."""
    gen = torch.Generator().manual_seed(seed)
    P, A, P2, F, F2 = [torch.as_tensor(V[k]) for k in ['P', 'A', 'P2', 'F', 'F2']]
    n = len(P)
    ES = torch.zeros(n); SE = torch.zeros(n); NLL = torch.zeros(n); HP = torch.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        z = enc(m, P[sl], F[sl]); y = enc(m, P2[sl], F2[sl]); a = A[sl]
        s = predict_samples(m, z, a, M, gen)
        ES[sl] = energy_score(s, y)
        mu = s.mean(0); SE[sl] = (mu - y).pow(2).sum(-1)
        var = s.var(0).clamp_min(1e-4) if m.kind != 'det' else None
        NLL[sl] = (0.5 * ((y - mu) ** 2 / var + var.log()).sum(-1)) if var is not None else 0.0
        fail = (((s - lm.anchor.z) ** 2).sum(-1) < lm.anchor.tau).float().mean(0)     # model's predicted P(fell)
        HP[sl] = fail
    return dict(ES=ES.numpy(), SE=SE.numpy(), NLL=NLL.numpy(), HP=HP.numpy())


# ------------------------------------------------------------------------------------------ true / imagined plan costs
def plan_cost_from_latents(lm, ztraj, zg):
    """Planner cost (LatentModel.cost) applied to a latent trajectory (B,H,D): same functional as the planner's scoring."""
    acc = ((ztraj - zg[:, None]) ** 2).sum(-1).mean(1)                 # mean_t ||z_t - zg||^2
    fin = ((ztraj[:, -1] - zg) ** 2).sum(-1)
    failed = ((((ztraj - lm.anchor.z) ** 2).sum(-1)) < lm.anchor.tau).float().max(1).values
    return (fin + lm.stage_w * acc) / lm.scale + lm.kappa * failed


@torch.no_grad()
def model_plan_cost(lm, z0, acts, zg, M=32, gen=None, chunk=2000):
    out = []
    for i in range(0, len(z0), chunk):
        sl = slice(i, i + chunk)
        feat = lm.rollout(z0[sl], acts[sl][:, None], M, gen, zg[sl])       # (B,1,M,D+2)
        out.append(lm.cost(feat, zg[sl])[:, 0].mean(-1))
    return torch.cat(out)


def waypoint_goals(p0, rng, dist=0.45):
    """Intermediate goals reachable within the horizon: a point `dist` from the start in the direction of a task goal."""
    gt = np.stack([rng.uniform(0.82, 0.94, len(p0)), rng.uniform(0.34, 0.46, len(p0))], 1)
    u = gt - p0
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
    return np.clip(p0 + dist * u, 0, 1).astype(np.float32)


def zone_mask(p0):
    return (p0[:, 0] > 0.05) & (p0[:, 0] < 0.55) & (p0[:, 1] > 0.30) & (p0[:, 1] < 0.60)


def plan_pool_stats(m, lm, W, cell=0.05, N=16, R=200, min_n=40, seed=0):
    """Replay-style, planner-aware statistics on logged plan pools.

    Windows starting in the same cell share one waypoint goal. For each cell we draw R random sub-pools of N logged plans,
    let the planner's scoring (model-predicted planner cost) pick its argmin, and look at the REALISED cost / fall of the picked
    logged plan.  Also within-cell Kendall tau and the optimism (realised minus predicted) of the picked plans.
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    zone = zone_mask(W['p0'])
    p0, acts, traj, fell = W['p0'][zone], W['acts'][zone], W['traj'][zone], W['fell'][zone]
    cid = np.floor(p0 / cell).astype(int)
    key = cid[:, 0] * 1000 + cid[:, 1]
    centres = (cid + 0.5) * cell
    # one goal per cell (deterministic per cell index so every model sees the same goals)
    goals = np.zeros_like(p0)
    for k in np.unique(key):
        sel = key == k
        goals[sel] = waypoint_goals(centres[sel][:1], np.random.default_rng(10_000 + int(k)))[0]
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    z0 = enc(m, t(p0))
    zg = enc(m, t(goals))
    zt = enc(m, t(traj), torch.as_tensor(fell))                           # (W,H,D)
    y = plan_cost_from_latents(lm, zt, zg).numpy()
    J = model_plan_cost(lm, z0, t(acts), zg, 32, gen).numpy()
    fall = fell.any(1).astype(float)
    res = dict(pick_y=[], pick_fall=[], pick_opt=[], mean_y=[], mean_fall=[], best_y=[], tau=[], tw=[], rmse=[], bias=[])
    from scipy.stats import kendalltau
    for k in np.unique(key):
        sel = np.where(key == k)[0]
        if len(sel) < min_n:
            continue
        yk, Jk, fk = y[sel], J[sel], fall[sel]
        res['tau'].append(kendalltau(Jk, yk)[0]); res['tw'].append(len(sel))
        res['rmse'].append(np.sqrt(np.mean((yk - Jk) ** 2))); res['bias'].append(np.mean(yk - Jk))
        for _ in range(R // 4):
            for_ = rng.choice(len(sel), size=N, replace=False)
            j = for_[np.argmin(Jk[for_])]
            res['pick_y'].append(yk[j]); res['pick_fall'].append(fk[j]); res['pick_opt'].append(yk[j] - Jk[j])
            res['mean_y'].append(yk[for_].mean()); res['mean_fall'].append(fk[for_].mean()); res['best_y'].append(yk[for_].min())
    out = {k: float(np.mean(v)) for k, v in res.items() if k not in ('tau', 'tw')}
    tw = np.array(res['tw']); out['tau_cell'] = float(np.average(res['tau'], weights=tw))
    out['pick_gain'] = out['pick_y'] - out['mean_y']
    out['pick_fall_gain'] = out['pick_fall'] - out['mean_fall']
    out['pick_regret'] = out['pick_y'] - out['best_y']
    out['n_cells'] = len(tw); out['n_windows'] = int(tw.sum())
    return out


def multistep_stats(m, lm, W, n=3000, M=16, seed=0):
    """H-step energy score (final latent) of open-loop rollouts along logged actions, all windows (plan-agnostic)."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(W['p0']), size=min(n, len(W['p0'])), replace=False)
    p0, acts, traj, fell = [torch.as_tensor(W[k][idx]) for k in ['p0', 'acts', 'traj', 'fell']]
    z = enc(m, p0)
    zs = z[None].expand(M, *z.shape).clone() if m.kind != 'det' else z[None].clone()
    for k in range(acts.shape[1]):
        a = acts[:, k][None].expand(zs.shape[0], *acts[:, k].shape)
        zs = m.pred(zs, a)[0] if m.kind == 'det' else m.sample_next(zs, a, gen=gen)
    y = enc(m, traj[:, -1], fell[:, -1])
    if zs.shape[0] == 1:
        return dict(ES_H=float((zs[0] - y).norm(dim=-1).mean()), SE_H=float((zs[0] - y).pow(2).sum(-1).mean()))
    es = energy_score(zs, y)
    return dict(ES_H=float(es.mean()), SE_H=float((zs.mean(0) - y).pow(2).sum(-1).mean()))


# ------------------------------------------------------------------------------------------ imagined closed loop + occupancy
@torch.no_grad()
def imagined_closed_loop(m, lm, E=24, steps=70, N=16, Mp=4, iters=2, seed=4242):
    """Run the SAME planner as the ground-truth evaluation but with the model as the environment (direct method).
    Returns imagined success / fall / timeout and the visited latent positions + actions (for occupancy weighting)."""
    env = StochNav(VARIANT, 1, 0)
    env.rng = np.random.default_rng(seed)
    s, g = env.sample_start_goal(E)
    gen = torch.Generator().manual_seed(seed)
    z = enc(m, torch.tensor(s, dtype=torch.float32)); zg = enc(m, torch.tensor(g, dtype=torch.float32))
    sf = NoiseAwareScorer(0.0, 0.1, False)
    alive = torch.ones(E, dtype=torch.bool); fall = torch.zeros(E, dtype=torch.bool); succ = torch.zeros(E, dtype=torch.bool)
    visits, vacts = [], []
    for t in range(steps):
        plan = cem(lm, z, zg, H, N, Mp, iters=iters, gen=gen, score_fn=sf)
        a = plan[:, 0]
        visits.append(z[alive].clone()); vacts.append(a[alive].clone())
        if m.kind == 'es':
            zn = m.sample_next(z, a, gen=gen)
        else:
            zn = m.sample_next(z, a, gen=gen)
        z = torch.where(alive[:, None], zn, z)
        f = (((z - lm.anchor.z) ** 2).sum(-1) < lm.anchor.tau)
        sc = (0.5 * (z[:, :2] - zg[:, :2]).norm(dim=-1) < 0.06) & ~f
        fall |= alive & f; succ |= alive & sc & ~f
        alive &= ~(fall | succ)
        if not alive.any():
            break
    V = torch.cat(visits); Av = torch.cat(vacts)
    return dict(im_succ=float(succ.float().mean()), im_fall=float(fall.float().mean()),
                im_timeout=float((~(succ | fall)).float().mean())), V.numpy(), Av.numpy()


def hist2d(xy, bins=20):
    """xy in latent coords [-1,1]^2 (identity encoder) -> cell index."""
    c = np.clip(((xy + 1) / 2 * bins).astype(int), 0, bins - 1)
    return c[:, 0] * bins + c[:, 1]


def occupancy_weights(vis_xy, data_xy, bins=20, smooth=1.0):
    q = np.bincount(hist2d(vis_xy, bins), minlength=bins * bins).astype(float) + 0.0
    p = np.bincount(hist2d(data_xy, bins), minlength=bins * bins).astype(float)
    q = (q + smooth * 0.0) / max(q.sum(), 1); p = (p + 1.0) / (p.sum() + bins * bins)
    w = q / p
    return w, hist2d(data_xy, bins)
