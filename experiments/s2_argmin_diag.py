"""S2 diagnostic linking Phase A and Phase B: does the pointwise band lose coverage for the argmin-selected member of a CEM population on the
closed-loop state distribution (chained CRN), and is the executed plan (mean of the elites) affected?

collect:  python experiments/s2_argmin_diag.py collect --ckpt suite_hi/ckpt/es_s0.pt --m 0 --seeds 1020 --N 32 --steps 60 --out results/s2/argmin_N32_m0.npz
analyze:  python experiments/s2_argmin_diag.py analyze --Ns 32,64
Per visited state (base planner closed loop, wind 0.09): the final population (N members) with planner scores s (8 chained-CRN particles), model fall fraction pf,
true-simulator cost L and fall probability pT (150 fresh particles) of EVERY member and of the executed (elite-mean) plan.
Analysis: pools of size n are sub-sampled from the final population (20 per state); the selected plan is the argmin of s over the pool. Pointwise band: quantile of
residuals of random pool members; selected band: quantile of residuals of the selected plans; both calibrated on half of the episodes and tested on the other half
(episode-level split, 5 splits); coverage is that of the selected plan. Functional: fall probability (R = pT - pf) and total planner cost (R = L - s).
"""
import argparse, sys, time, glob
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm import s2_scc as S
from selwm.noise_aware import NoiseAwareScorer

def collect(a):
    assert a.seeds >= 1000
    mem, anchor, lm, lm_ind = S.load_models(a.ckpt)
    E = 16
    env = S.SlotEnv(a.seeds, E, wind=0.09, phase=0)
    gen = torch.Generator().manual_seed(a.seeds); gen_t = torch.Generator().manual_seed(a.seeds + 500_000)
    sc = NoiseAwareScorer(0.0, 0.1, False)
    R = {k: [] for k in ['s', 'pf', 'L', 'pT', 'Le', 'pTe', 'se', 'pfe', 'slot', 'jep']}
    prev = np.zeros(E, bool); t0 = time.time()
    for t in range(a.steps):
        p = torch.tensor(env.p, dtype=torch.float32); fell = torch.tensor(env.fell)
        z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32))
        lm.begin_replan(reset_mask=prev)
        pop = S.cem_pop(lm, z0, zg, a.N, 8, 3, gen, sc)
        se, pfe = S.score_exec(lm, z0, zg, pop['plan'], 8, gen)
        plans = torch.cat([pop['acts'], pop['plan'][:, None]], 1)
        L, pT = S.true_cost_w(lm, mem, anchor, p, zg, plans, a.Mt, gen_t, 0.09)
        for k, v in (('s', pop['s']), ('pf', pop['pf']), ('L', L[:, :-1]), ('pT', pT[:, :-1]), ('Le', L[:, -1]), ('pTe', pT[:, -1]), ('se', se), ('pfe', pfe)):
            R[k].append(v.numpy())
        R['slot'].append(np.arange(E)); R['jep'].append(env.j.copy())
        env.step(pop['plan'][:, 0].numpy(), np.arange(E)); fin = env.finished(np.arange(E)); prev = fin.copy()
        for i in np.where(fin)[0]: env.next_episode(i)
        if (t + 1) % 20 == 0: print(t + 1, f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(a.out, **{k: np.concatenate(v) for k, v in R.items()})

def cq(R, alpha):
    R = np.sort(np.ravel(R)); n = len(R); k = int(np.ceil((1 - alpha) * (n + 1)))
    return R[k - 1] if k <= n else np.inf

def analyze(a):
    out = []
    P = lambda s='': (print(s), out.append(s))
    for N in [int(x) for x in a.Ns.split(',')]:
        files = sorted(glob.glob(f'results/s2/argmin_N{N}_m*.npz'))
        if not files: continue
        D = [dict(np.load(f)) for f in files]
        P(f'Final population of the base CEM, N={N}, closed-loop states: ' + ', '.join(f"{len(d['s'])} states (model {i})" for i, d in enumerate(D)))
        P(f"{'pool n':>6s} {'functional':>18s} {'alpha':>5s} {'pointwise cov':>18s} {'selected-plan cov':>20s} {'pointwise q':>11s} {'selected q':>10s}")
        rng = np.random.default_rng(7)
        pools = [n for n in (2, 4, 8, 16, 32, 64) if n <= N]
        for n in pools:
            for fname in ('fall probability', 'total cost'):
                for alpha in (0.05, 0.10):
                    cp, cs, qp_, qs_ = [], [], [], []
                    for rep in range(5):
                        hp = hs = nt = 0; qpp = []; qss = []
                        for d in D:
                            S_, PF, PT, L_ = d['s'], d['pf'], d['pT'], d['L']
                            res = (PT - PF) if fname == 'fall probability' else (L_ - S_)
                            ne = len(S_)
                            idx = np.argsort(rng.random((ne, N)), 1)[:, :n]
                            sub_s = np.take_along_axis(S_, idx, 1); sub_r = np.take_along_axis(res, idx, 1)
                            sel = sub_s.argmin(1)[:, None]; r_sel = np.take_along_axis(sub_r, sel, 1)[:, 0]
                            ids = d['slot'] * 100000 + d['jep']; u = np.unique(ids); rng.shuffle(u)
                            cal_set = set(u[:len(u) // 2].tolist()); cal = np.array([i in cal_set for i in ids])
                            qp = cq(sub_r[cal], alpha); qs = cq(r_sel[cal], alpha)
                            hp += np.sum(r_sel[~cal] <= qp); hs += np.sum(r_sel[~cal] <= qs); nt += (~cal).sum(); qpp.append(qp); qss.append(qs)
                        cp.append(hp / nt); cs.append(hs / nt); qp_.append(np.mean(qpp)); qs_.append(np.mean(qss))
                    P(f'{n:6d} {fname:>18s} {alpha:5.2f} {np.mean(cp):8.3f} [{np.min(cp):.3f},{np.max(cp):.3f}] {np.mean(cs):9.3f} [{np.min(cs):.3f},{np.max(cs):.3f}] {np.mean(qp_):11.3f} {np.mean(qs_):10.3f}')
        P('executed plan (mean of the elites) for comparison, calibrated on random final-population members (pointwise) vs on the executed plans (selected):')
        for fname in ('fall probability', 'total cost'):
            for alpha in (0.05, 0.10):
                cp, cs = [], []
                for rep in range(5):
                    hp = hs = nt = 0
                    for d in D:
                        res_m = (d['pT'] - d['pf']) if fname == 'fall probability' else (d['L'] - d['s'])
                        res_e = (d['pTe'] - d['pfe']) if fname == 'fall probability' else (d['Le'] - d['se'])
                        ids = d['slot'] * 100000 + d['jep']; u = np.unique(ids); rng.shuffle(u)
                        cal_set = set(u[:len(u) // 2].tolist()); cal = np.array([i in cal_set for i in ids])
                        qp = cq(res_m[cal], alpha); qs = cq(res_e[cal], alpha)
                        hp += np.sum(res_e[~cal] <= qp); hs += np.sum(res_e[~cal] <= qs); nt += (~cal).sum()
                    cp.append(hp / nt); cs.append(hs / nt)
                P(f"  {fname:>18s} alpha {alpha:.2f}: pointwise {np.mean(cp):.3f} selected {np.mean(cs):.3f}")
        P()
    open('results/s2/argmin_diag.txt', 'w').write('\n'.join(out) + '\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode')
    ap.add_argument('--ckpt'); ap.add_argument('--m', type=int); ap.add_argument('--seeds', type=int); ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--steps', type=int, default=60); ap.add_argument('--Mt', type=int, default=150); ap.add_argument('--out'); ap.add_argument('--Ns', default='32,64')
    a = ap.parse_args()
    collect(a) if a.mode == 'collect' else analyze(a)
