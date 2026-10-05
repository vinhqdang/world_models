"""N3 tiny CPU check (1 thread): plan-level winner's curse and pointwise vs pool-uniform calibration on StochNav cliff_hi.

For a pool of N0 open-loop candidate plans per state we compute
  J8   : model score with M=8 particles (what the planner sees)
  J64  : model score with M=64 particles (model expectation, almost noise free)
  Jtrue: the same cost functional evaluated on TRUE-simulator particles (encoded by the same frozen encoder), Mt particles.
Then, for sub-pools of size N, we measure optimism of the selected plan, regret, and the coverage of
  (a) pointwise-calibrated band: q = 90% quantile of residual r = Jtrue - J8 over all (state, candidate) pairs,
  (b) selected-plan conformal band: q = 90% quantile of r at the selected plan over calibration states,
  (c) pool-uniform band: q = 90% quantile over calibration states of max_i r_i over the sub-pool.
Coverage = fraction of held-out states where Jtrue(selected) <= J8(selected) + q.

python results/n3/selection_check.py  (about a few minutes on one CPU thread)
"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, LatentModel, observe

CK = sys.argv[1] if len(sys.argv) > 1 else 'suite_hi/ckpt/es_s0.pt'
NS, N0, H, MT = int(sys.argv[2]) if len(sys.argv) > 2 else 240, 64, 10, 200
variant = 'cliff_hi'
ck = torch.load(CK, map_location='cpu'); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel'))
mem.load_state_dict(ck['state']); mem.eval()
anchor = FailureAnchor(mem.encode, variant, mem.obs, device='cpu')
e0 = StochNav(variant, 1, 4242); sp, gp = e0.sample_start_goal(512)
with torch.no_grad():
    zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs))
    zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
scale = float(((zs - zg_) ** 2).sum(-1).median())
lm = LatentModel(mem, variant, stochastic=True, crn=False, stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)

rng = np.random.default_rng(12345)
gen = torch.Generator().manual_seed(7)
env = StochNav(variant, 1, 0)

# ---- states: along the start->goal line, y jittered, outside the pit
s, g = env.sample_start_goal(NS)
t = rng.uniform(0.0, 0.7, (NS, 1))
P0 = s + t * (g - s); P0[:, 1] += rng.uniform(-0.04, 0.10, NS)
P0 = np.clip(P0, 0.02, 0.98)
bad = env._in(P0, env.pit); P0[bad, 1] = 0.33
P0 = P0.astype(np.float32); G = g.astype(np.float32)

# ---- candidate pool: goal-directed with correlated noise and a lateral bias
d = G - P0; d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
acts = np.zeros((NS, N0, H, 2), np.float32)
for i in range(NS):
    ar = np.zeros((N0, H, 2)); x = rng.normal(0, 0.7, (N0, 2))
    for k in range(H):
        x = 0.7 * x + 0.7 * rng.normal(0, 0.7, (N0, 2)); ar[:, k] = x
    bias = rng.uniform(-1, 1, (N0, 1)) * (rng.random((N0, 1)) < 0.6)
    base = np.broadcast_to(d[i], (N0, H, 2)).copy(); base[..., 1] += bias
    acts[i] = np.clip(0.8 * base + 0.5 * ar, -1, 1)
acts_t = torch.tensor(acts)

p0 = torch.tensor(P0); gt = torch.tensor(G)
with torch.no_grad():
    z0 = lm.obs_to_latent(p0, torch.zeros(NS, dtype=torch.bool)); zg = lm.obs_to_latent(gt)

def model_scores(M):
    out = []
    for lo in range(0, NS, 40):
        sl = slice(lo, lo + 40)
        feat = lm.rollout(z0[sl], acts_t[sl], M, gen, zg[sl])
        out.append(lm.cost(feat, zg[sl]))
    return torch.cat(out)                                   # (NS,N0,M)

t0 = time.time()
c8 = model_scores(8); c64 = model_scores(64)
J8, J64 = c8.mean(-1).numpy(), c64.mean(-1).numpy(); SE8 = (c8.std(-1) / np.sqrt(8)).numpy()
print('model scores', time.time() - t0, flush=True)

# ---- true-simulator particles, same cost functional
pit = torch.tensor(env.pit, dtype=torch.float32)
def in_pit(p):
    x, y = p[..., 0:1], p[..., 1:2]
    return ((x >= pit[:, 0]) & (x <= pit[:, 2]) & (y >= pit[:, 1]) & (y <= pit[:, 3])).any(-1)
wind = env.wind_base; step = env.step_size
def true_scores(lo, hi):
    n = hi - lo
    p = p0[lo:hi, None, None, :].expand(n, N0, MT, 2).clone()
    stuck = torch.zeros(n, N0, MT, dtype=torch.bool)
    acc = torch.zeros(n, N0, MT); failed = torch.zeros(n, N0, MT)
    g_ = zg[lo:hi]
    for k in range(H):
        a_ = acts_t[lo:hi, :, k, None, :].clamp(-1, 1)
        pn = p + step * a_
        sign = (torch.randint(0, 2, (n, N0, MT), generator=gen) * 2 - 1).float()
        pn[..., 1] = pn[..., 1] + wind * sign
        pn = pn.clamp(0, 1)
        pn = torch.where(stuck[..., None], p, pn)
        stuck = stuck | in_pit(pn)
        p = pn
        with torch.no_grad():
            z = mem.encode(observe(p, stuck, variant, mem.obs))
        acc = acc + ((z - g_[:, None, None, :]) ** 2).sum(-1) / H
        failed = torch.maximum(failed, (((z - anchor.z) ** 2).sum(-1) < anchor.tau).float())
    feat = torch.cat([z, acc[..., None], failed[..., None]], -1)
    return lm.cost(feat, g_), stuck.float().mean(-1)
Jt, pf = [], []
for lo in range(0, NS, 20):
    c, f = true_scores(lo, min(NS, lo + 20)); Jt.append(c.mean(-1)); pf.append(f)
Jt = torch.cat(Jt).numpy(); PF = torch.cat(pf).numpy()
print('true scores', time.time() - t0, flush=True)

# ---- analyses
cal = np.arange(NS) < NS // 2; tst = ~cal
R = Jt - J8                                                   # residual (positive = model optimistic)
out = dict(ckpt=CK, NS=NS, N0=N0, MT=MT, mean_model_gap_all=float(R.mean()), mean_gap_vs_J64=float((Jt - J64).mean()),
           mean_particle_noise_sd=float(np.sqrt(((J8 - J64) ** 2).mean())), mean_true_fail_prob=float(PF.mean()), rows=[])
rs = np.random.default_rng(0)
q_pt = float(np.quantile(R[cal], 0.9))
for N in (2, 4, 8, 16, 32, 64):
    reps = 30
    opt, reg, reg_J64, sel_cov = [], [], [], {'pointwise': [], 'selected': [], 'uniform': []}
    # draw subpool index sets once per (state, rep)
    idx = np.stack([[rs.choice(N0, N, replace=False) for _ in range(reps)] for _ in range(NS)])   # (NS,reps,N)
    j8 = np.take_along_axis(J8[:, None, :].repeat(reps, 1), idx, 2)
    j64 = np.take_along_axis(J64[:, None, :].repeat(reps, 1), idx, 2)
    jt = np.take_along_axis(Jt[:, None, :].repeat(reps, 1), idx, 2)
    se8 = np.take_along_axis(SE8[:, None, :].repeat(reps, 1), idx, 2)
    sel = j8.argmin(-1)
    gs = lambda arr, s_: np.take_along_axis(arr, s_[..., None], 2)[..., 0]
    r_sel = gs(jt, sel) - gs(j8, sel)                          # residual at selected plan (NS,reps)
    bias_part = gs(j64, sel) - gs(j8, sel)                     # (J64 - J8) at the selected plan: particle-noise optimism
    out_row = dict(N=N, optimism=float(r_sel.mean()), particle_noise_part=float(bias_part.mean()),
                   model_bias_part=float((gs(jt, sel) - gs(j64, sel)).mean()),
                   regret_vs_best_in_pool=float((gs(jt, sel) - jt.min(-1)).mean()),
                   true_cost_selected=float(gs(jt, sel).mean()))
    # selection by J64 (8x particles) for reference
    sel64 = j64.argmin(-1)
    out_row['regret_J64_select'] = float((gs(jt, sel64) - jt.min(-1)).mean())
    # pessimistic selection by J8 + beta*SE
    for beta in (1.0, 2.0):
        sb = (j8 + beta * se8).argmin(-1); out_row[f'regret_pess_beta{beta}'] = float((gs(jt, sb) - jt.min(-1)).mean())
    # coverage of the selected plan under the three bands (calibrate on cal states, test on tst states)
    q_sel = float(np.quantile(r_sel[cal], 0.9))
    r_all = jt - j8
    q_uni = float(np.quantile(r_all[cal].max(-1), 0.9))
    q_pt_N = float(np.quantile(r_all[cal], 0.9))
    out_row.update(q_pointwise=q_pt_N, q_selected=q_sel, q_uniform=q_uni,
                   cov_pointwise=float((r_sel[tst] <= q_pt_N).mean()), cov_selected=float((r_sel[tst] <= q_sel).mean()),
                   cov_uniform=float((r_sel[tst] <= q_uni).mean()),
                   # statistic of interest: how often does the uniform band cover ALL pool members (its own guarantee)
                   cov_uniform_all=float((r_all[tst].max(-1) <= q_uni).mean()))
    out['rows'].append(out_row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out_row.items()}, flush=True)
json.dump(out, open('results/n3/selection_check_%s.json' % CK.split('/')[-1].replace('.pt', ''), 'w'), indent=1)
print('done', time.time() - t0)
