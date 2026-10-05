"""S1 Phase A analysis: selected-plan coverage of pointwise / selected-plan-conformal / pool-max bands on real CEM populations.

python experiments/s1_analyze.py --scheme indep
Residual R = true cost (200 true particles, set A) - J8 (planner score). Bands are one-sided: covered iff R_sel <= q.
Splits: 5 random 500/500 calibration/test splits of the 1000 states per checkpoint; sub-pools: 20 random draws per state and N.
Pooled = calibration sets of the 3 checkpoints concatenated (1500 cal / 1500 test states), one band for all.
"""
import argparse, json, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', default='indep'); ap.add_argument('--alpha', type=float, default=0.10)
ap.add_argument('--reps', type=int, default=20); ap.add_argument('--splits', type=int, default=5)
ap.add_argument('--ks', default='0,1,2')
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(',')]
D = {k: np.load(f'results/s1/pops_es_s{k}_{args.scheme}.npz') for k in KS}
ALPHA = args.alpha


def cq(x, alpha=ALPHA):
    """split-conformal upper quantile: ceil((1-alpha)(n+1))-th smallest."""
    x = np.sort(np.asarray(x).ravel()); n = len(x)
    k = min(n, int(math.ceil((1 - alpha) * (n + 1))))
    return x[k - 1]


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def subpools(rng, n, N0, N):
    return np.argsort(rng.random((n, N0)), 1)[:, :N]


def take(a, idx):
    return np.take_along_axis(a, idx, 1)


def analyze(pop, N, ks, seed=0):
    """Returns dict of metrics for the list of checkpoints `ks` combined (bands calibrated on the pooled calibration set)."""
    rng = np.random.default_rng(1000 * N + seed)
    acc = {m: [] for m in ['cov_pt', 'cov_sel', 'cov_max', 'q_pt', 'q_sel', 'q_max', 'cov_pt_member', 'cov_max_all', 'cov_sel_all_members']}
    st = {m: [] for m in ['opt', 'opt_noise', 'opt_bias', 'reg_naive', 'reg_cf', 'reg_cf_J64', 'jt_sel', 'tfp_sel']}
    sel_store = []
    for rep in range(args.reps):
        per = []
        for k in ks:
            d = D[k]; J8, Jt, JtB, J64 = d[f'{pop}_J8'], d[f'{pop}_JtA'], d[f'{pop}_JtB'], d[f'{pop}_J64']
            n, N0 = J8.shape
            idx = subpools(rng, n, N0, N)
            j8, jt, jb, j64 = take(J8, idx), take(Jt, idx), take(JtB, idx), take(J64, idx)
            sel = j8.argmin(1)[:, None]
            g = lambda a: np.take_along_axis(a, sel, 1)[:, 0]
            r = jt - j8
            per.append(dict(r=r, rsel=g(r), rmax=r.max(1)))
            st['opt'].append(g(r)); st['opt_noise'].append(g(j64) - g(j8)); st['opt_bias'].append(g(jt) - g(j64))
            st['reg_naive'].append(g(jt) - jt.min(1))
            orc = jb.argmin(1)[:, None]; st['reg_cf'].append(g(jt) - np.take_along_axis(jt, orc, 1)[:, 0])
            s64 = j64.argmin(1)[:, None]
            st['reg_cf_J64'].append(np.take_along_axis(jt, s64, 1)[:, 0] - np.take_along_axis(jt, orc, 1)[:, 0])
            st['jt_sel'].append(g(jt))
        for s in range(args.splits):
            cal_r, cal_sel, cal_max, t_sel, t_r, t_max = [], [], [], [], [], []
            for p_, k in zip(per, ks):
                n = len(p_['rsel'])
                perm = np.random.default_rng([s, k, 77]).permutation(n)          # same splits across N and reps
                c, t = perm[: n // 2], perm[n // 2:]
                cal_r.append(p_['r'][c].ravel()); cal_sel.append(p_['rsel'][c]); cal_max.append(p_['rmax'][c])
                t_sel.append(p_['rsel'][t]); t_r.append(p_['r'][t]); t_max.append(p_['rmax'][t])
            qp, qs, qm = cq(np.concatenate(cal_r)), cq(np.concatenate(cal_sel)), cq(np.concatenate(cal_max))
            ts, tr, tm = np.concatenate(t_sel), np.concatenate(t_r), np.concatenate(t_max)
            acc['cov_pt'].append((ts <= qp).mean()); acc['cov_sel'].append((ts <= qs).mean()); acc['cov_max'].append((ts <= qm).mean())
            acc['q_pt'].append(qp); acc['q_sel'].append(qs); acc['q_max'].append(qm)
            acc['cov_pt_member'].append((tr <= qp).mean()); acc['cov_max_all'].append((tm <= qm).mean())
            acc['cov_sel_all_members'].append(0.0)
    nt = 500 * len(ks)
    out = dict(N=N, n_test=nt)
    for m in ['cov_pt', 'cov_sel', 'cov_max', 'cov_pt_member', 'cov_max_all']:
        p = float(np.mean(acc[m])); lo, hi = wilson(p, nt)
        out[m] = p; out[m + '_ci'] = [lo, hi]
        out[m + '_split_sd'] = float(np.std(acc[m]))
    for m in ['q_pt', 'q_sel', 'q_max']:
        out[m] = float(np.mean(acc[m]))
    for m, v in st.items():
        if v: out[m] = float(np.concatenate(v).mean()); out[m + '_se'] = float(np.concatenate(v).std() / math.sqrt(len(np.concatenate(v)) / args.reps))
    return out


NS_BY_POP = {'f32': [2, 4, 8, 16, 32], 'i32': [2, 4, 8, 16, 32], 'f64': [2, 4, 8, 16, 32, 64]}
res = {'scheme': args.scheme, 'alpha': ALPHA, 'reps': args.reps, 'splits': args.splits, 'tables': {}}
for pop, Ns in NS_BY_POP.items():
    for name, ks in [(f's{k}', [k]) for k in KS] + [('pooled', KS)]:
        res['tables'][f'{pop}/{name}'] = [analyze(pop, N, ks) for N in Ns]
# descriptive stats of the populations
desc = {}
for pop in NS_BY_POP:
    for k in KS:
        d = D[k]; J8, Jt, pf, J64 = d[f'{pop}_J8'], d[f'{pop}_JtA'], d[f'{pop}_pfA'], d[f'{pop}_J64']
        desc[f'{pop}/s{k}'] = dict(J8_mean=float(J8.mean()), J64_mean=float(J64.mean()), Jt_mean=float(Jt.mean()), mean_true_fail=float(pf.mean()),
                                   resid_mean=float((Jt - J8).mean()), resid_sd=float((Jt - J8).std()), rms_J8_minus_J64=float(np.sqrt(((J8 - J64) ** 2).mean())),
                                   within_state_sd_Jt=float(Jt.std(1).mean()), within_state_sd_J8=float(J8.std(1).mean()),
                                   corr_resid_within_state=float(np.mean([np.corrcoef(J8[i], Jt[i])[0, 1] for i in range(len(J8))])))
res['desc'] = desc
json.dump(res, open(f'results/s1/phaseA_{args.scheme}.json', 'w'), indent=1)

# ---- text tables
L = []
def row(o):
    f = lambda m: f"{o[m]:.3f} [{o[m + '_ci'][0]:.2f},{o[m + '_ci'][1]:.2f}]"
    return (f"| {o['N']} | {f('cov_pt')} | {f('cov_sel')} | {f('cov_max')} | {o['q_pt']:.2f} / {o['q_sel']:.2f} / {o['q_max']:.2f} | "
            f"{o['opt']:+.3f} ({o['opt_noise']:+.2f}/{o['opt_bias']:+.2f}) | {o['jt_sel']:.3f} | {o['reg_naive']:.3f} / {o['reg_cf']:.3f} / {o['reg_cf_J64']:.3f} |")
for key, tab in res['tables'].items():
    L.append(f"### {key}  (scheme={args.scheme}, nominal {1 - ALPHA:.2f}, n_test per cell={tab[0]['n_test']})")
    L.append('| N | cov pointwise [Wilson95] | cov selected-conf | cov pool-max | width pt / sel / max | optimism mean R_sel (particle-noise part / model-bias part) | true cost of selected | regret naive / crossfit / crossfit-J64sel |')
    L.append('|---|---|---|---|---|---|---|---|')
    L += [row(o) for o in tab]; L.append('')
L.append('### population descriptives'); L.append(json.dumps(desc, indent=1))
open(f'results/s1/phaseA_{args.scheme}.txt', 'w').write('\n'.join(L))
print('\n'.join(L[:60]))
