"""Summaries for D8. Reads results/d8/<arm>_m<model>_e<evalseed>.json; writes results/d8/summary_table.txt and summary.json.
Per arm: pooled success/fall/timeout with stratified (by run) episode bootstrap 95% CI, median steps-to-goal with bootstrap CI.
Paired comparison vs the baseline arm on matched (model seed, eval seed, slot, episode index) episodes (same start/goal and wind
stream over time steps; planner randomness and trajectories differ): paired bootstrap of rate differences and exact McNemar test.
Unpaired: two-proportion z test on all episodes."""
import json, glob, re, sys, collections
import numpy as np
from scipy import stats

R = np.random.default_rng(12345); B = 4000
files = sorted(glob.glob('results/d8/*_m*_e*.json'))
runs = collections.defaultdict(dict)        # arm -> (m,e) -> record list
meta = {}; RT = {}
MAIN = ('ol32', 'ol36', 'fbs32')
for f in files:
    mm = re.match(r'results/d8/(.+)_m(\d)_e(\d+)\.json', f)
    if not mm: continue
    arm, m, e = mm.group(1), int(mm.group(2)), int(mm.group(3))
    d = json.load(open(f)); runs[arm][(m, e)] = {(s, j): (o, st) for s, j, o, st in d['records']}
    meta.setdefault(arm, dict(ps=d['particle_steps_per_env_step'], sec=[], args=d['args'], K=[]))
    meta[arm]['sec'].append(d['sec']); RT[(arm, m, e)] = d['args']['total_steps']; meta[arm]['steps'] = meta[arm].get('steps', 0) + d['args']['total_steps']
    if d.get('K_mean'): meta[arm]['K'].append(d['K_mean'])

def ends(r):
    """end step of each episode (slot, j) from cumulative durations per slot"""
    cum = collections.defaultdict(int); out = {}
    for (s, j) in sorted(r): cum[s] += r[(s, j)][1]; out[(s, j)] = cum[s]
    return out

def pooled(arm, keys=None, T=None, start_cut=False):
    """start_cut: keep only episodes that STARTED early enough (start <= run length - ep_len) to be certain to finish inside the window,
    which removes the length bias of counting only completed episodes in a finite streaming window."""
    out = []
    for k, r in runs[arm].items():
        if keys is not None and k not in keys: continue
        en = ends(r) if (T is not None or start_cut) else None
        Tk = RT[(arm, k[0], k[1])] if T is None else min(T, RT[(arm, k[0], k[1])])
        out += [(k, o, st) for (sj, (o, st)) in r.items() if (T is None or en[sj] <= Tk) and (not start_cut or en[sj] - st <= Tk - 120)]
    return out

def boot_rates(arm):
    strata = collections.defaultdict(list)
    for k, o, st in pooled(arm): strata[k].append((o, st))
    arrs = {k: (np.array([o == 'success' for o, _ in v]), np.array([o == 'fall' for o, _ in v]), np.array([o == 'timeout' for o, _ in v]),
                np.array([st if o == 'success' else np.nan for o, st in v], float)) for k, v in strata.items()}
    n = sum(len(v[0]) for v in arrs.values())
    pt = [sum(v[i].sum() for v in arrs.values()) / n for i in range(3)]
    allsteps = np.concatenate([v[3] for v in arrs.values()]); allsteps = allsteps[~np.isnan(allsteps)]
    pm = float(np.median(allsteps))
    bs = np.zeros((B, 4))
    for b in range(B):
        tot = np.zeros(3); st_b = []
        for v in arrs.values():
            idx = R.integers(0, len(v[0]), len(v[0]))
            tot += [v[0][idx].sum(), v[1][idx].sum(), v[2][idx].sum()]
            s = v[3][idx]; st_b.append(s[~np.isnan(s)])
        bs[b, :3] = tot / n; bs[b, 3] = np.median(np.concatenate(st_b))
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    return n, pt + [pm], ci

def paired_keys(a, b):
    """matched episodes: for each (m,e) run and slot, episode indices below the minimum completed count over both arms."""
    out = []
    for k in runs[a]:
        if k not in runs[b]: continue
        slots = collections.defaultdict(lambda: [0, 0])
        for (s, j) in runs[a][k]: slots[s][0] = max(slots[s][0], j + 1)
        cb = collections.defaultdict(int)
        for (s, j) in runs[b][k]: cb[s] = max(cb[s], j + 1)
        for s, (na, _) in slots.items():
            for j in range(min(na, cb.get(s, 0))): out.append((k, s, j))
    return out

def paired(a, b):
    """b minus a"""
    ks = paired_keys(a, b)
    oa = np.array([runs[a][k][(s, j)][0] for k, s, j in ks]); ob = np.array([runs[b][k][(s, j)][0] for k, s, j in ks])
    res = dict(n=len(ks))
    strata = np.array([hash(k) for k, s, j in ks])
    for name in ('success', 'fall', 'timeout'):
        xa, xb = (oa == name), (ob == name)
        diff = xb.mean() - xa.mean()
        bs = []
        for _ in range(B):
            idx = R.integers(0, len(ks), len(ks)); bs.append(xb[idx].mean() - xa[idx].mean())
        n01 = int((~xa & xb).sum()); n10 = int((xa & ~xb).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        res[name] = (diff, *np.percentile(bs, [2.5, 97.5]), p, n01, n10)
    return res

def common(a, b):
    """main arms: all runs of each arm, own windows; otherwise common runs and the shorter window"""
    if a in MAIN and b in MAIN: return None, None
    keys = set(runs[a]) & set(runs[b])
    return keys, min(max(RT[(a, *k)] for k in keys), max(RT[(b, *k)] for k in keys))

def unpaired(a, b):
    """b minus a; the arm with more runs / steps is restricted to the other's runs and step window"""
    res = {}
    keys, T = common(a, b)
    pa, pb = pooled(a, keys, T), pooled(b, keys, T)
    res['n'] = (len(pa), len(pb))
    for i, name in enumerate(('success', 'fall', 'timeout')):
        xa = np.array([o == name for _, o, _ in pa]); xb = np.array([o == name for _, o, _ in pb])
        p0 = (xa.sum() + xb.sum()) / (len(xa) + len(xb))
        se = np.sqrt(p0 * (1 - p0) * (1 / len(xa) + 1 / len(xb)))
        z = (xb.mean() - xa.mean()) / se if se > 0 else 0
        res[name] = (xb.mean() - xa.mean(), 2 * stats.norm.sf(abs(z)), xa.mean(), xb.mean())
    sa = [st for _, o, st in pa if o == 'success']; sb = [st for _, o, st in pb if o == 'success']
    res['steps_mw_p'] = stats.mannwhitneyu(sa, sb).pvalue
    return res

order = [a for a in ['ol32', 'ol36', 'fbs32', 'fbs32_lam05', 'fbs32_M12', 'fbs32_k35', 'fbc32'] if a in runs] + sorted(a for a in runs if a not in ['ol32', 'ol36', 'fbs32', 'fbs32_lam05', 'fbs32_M12', 'fbs32_k35', 'fbc32'])
L = []; J = {}
L.append('Per-arm pooled results (cliff_hi, learned energy-score models seeds 0-2, E=16, H=10, iters=3, 95%% CI = stratified episode bootstrap, B=%d)' % B)
L.append(f"{'arm':14s} {'runs':>4s} {'eps':>5s} {'success':>22s} {'fall':>22s} {'timeout':>22s} {'median steps (CI)':>20s} {'part-steps/step':>15s} {'sec/run':>8s}  per-model-seed success")
for arm in order:
    n, pt, ci = boot_rates(arm)
    ps = {}
    for k in runs[arm]:
        v = [o for o, _ in runs[arm][k].values()]; ps.setdefault(k[0], []).append(np.mean([o == 'success' for o in v]))
    L.append(f"{arm:14s} {len(runs[arm]):4d} {n:5d} " + ' '.join(f"{pt[i]:.3f} [{ci[0][i]:.3f},{ci[1][i]:.3f}]".rjust(22) for i in range(3)) +
             f" {pt[3]:6.1f} [{ci[0][3]:.1f},{ci[1][3]:.1f}]".rjust(21) + f" {meta[arm]['ps']:15d} {np.mean(meta[arm]['sec']):8.0f}  " + ' '.join(f"m{m}:{np.mean(v):.3f}" for m, v in sorted(ps.items())))
    J[arm] = dict(episodes=n, success=pt[0], fall=pt[1], timeout=pt[2], median_steps=pt[3], ci_success=[ci[0][0], ci[1][0]], ci_fall=[ci[0][1], ci[1][1]],
                  ci_timeout=[ci[0][2], ci[1][2]], ci_median_steps=[ci[0][3], ci[1][3]], particle_steps_per_env_step=meta[arm]['ps'], K_mean=np.mean(meta[arm]['K'], 0).tolist() if meta[arm]['K'] else None)
L.append('')
L.append('Comparisons (difference = arm minus baseline; paired = matched (model seed, eval seed, slot, episode idx); McNemar exact p)')
for base in ['ol32', 'ol36', 'fbs32']:
    for arm in order:
        if arm == base or base not in runs: continue
        if base == 'ol36' and arm == 'ol32': continue
        if base == 'fbs32' and arm in MAIN: continue
        P = paired(base, arm); U = unpaired(base, arm)
        L.append(f"-- {arm} vs {base}: paired episodes n={P['n']}; unpaired n={U['n'][1]} vs {U['n'][0]} (common runs, common step window)")
        for name in ('success', 'fall', 'timeout'):
            d, lo, hi, p, n01, n10 = P[name]
            L.append(f"   {name:8s} paired diff {d:+.3f} [{lo:+.3f},{hi:+.3f}] McNemar p={p:.4f} (discordant {n01}/{n10});  unpaired diff {U[name][0]:+.3f} ({U[name][2]:.3f} -> {U[name][3]:.3f}) z-test p={U[name][1]:.4f}")
        L.append(f"   steps-to-goal (successes) Mann-Whitney p={U['steps_mw_p']:.4f}")
        J[f'{arm}_vs_{base}'] = dict(paired=P, unpaired=U)
L.append('')
L.append('Robustness: length-bias-free subset (only episodes that started at or before step total_steps-120, so every one finishes inside the window); unpaired two-proportion z tests, bootstrap CI as above')
for arm in order:
    pp = pooled(arm, start_cut=True); n = len(pp)
    x = {nm: np.array([o == nm for _, o, _ in pp]) for nm in ('success', 'fall', 'timeout')}
    bs = np.array([[x[nm][R.integers(0, n, n)].mean() for nm in x] for _ in range(1000)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    L.append(f"{arm:14s} eps={n:4d} " + ' '.join(f"{nm} {x[nm].mean():.3f} [{ci[0][i]:.3f},{ci[1][i]:.3f}]" for i, nm in enumerate(x)))
    J[arm]['startcut'] = dict(n=n, **{nm: float(x[nm].mean()) for nm in x})
for base in ['ol32', 'ol36', 'fbs32']:
    for arm in order:
        if arm == base or base not in runs or (base == 'ol36' and arm == 'ol32') or (base == 'fbs32' and arm in MAIN): continue
        keys, T = common(base, arm)
        pa, pb = pooled(base, keys, T, True), pooled(arm, keys, T, True)
        out = []
        for nm in ('success', 'fall', 'timeout'):
            xa = np.array([o == nm for _, o, _ in pa]); xb = np.array([o == nm for _, o, _ in pb])
            p0 = (xa.sum() + xb.sum()) / (len(xa) + len(xb)); se = np.sqrt(p0 * (1 - p0) * (1 / len(xa) + 1 / len(xb)))
            z = (xb.mean() - xa.mean()) / se if se > 0 else 0
            out.append(f"{nm} {xb.mean() - xa.mean():+.3f} (p={2 * stats.norm.sf(abs(z)):.4f})")
        L.append(f"   {arm} vs {base} (n={len(pb)} vs {len(pa)}): " + '; '.join(out))
txt = '\n'.join(L); print(txt)
open('results/d8/summary_table.txt', 'w').write(txt + '\n'); json.dump(J, open('results/d8/summary.json', 'w'), indent=1, default=float)
