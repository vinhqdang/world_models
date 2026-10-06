"""S2 Phase B analysis. Reads results/s2/runs/{cond}_{arm}_m{m}_j{j0}.json (+ .steps.npz), writes results/s2/summary.txt and summary.json.

cond: in (wind 0.09) | sh (wind 0.09 -> 0.13; episodes counted are those of the post-shift phase)
Per arm: pooled success / fall / timeout with bootstrap 95% CI (resampling episodes within model-seed strata, 4000 draws), median steps of successes,
predictor rows per env step; for SCC arms also commit / escalate / defer rates and the diagnostics from the true simulator.
Paired comparisons use episodes matched on (model seed, slot, episode index): same start/goal and same wind sequence, different planner randomness
and trajectories (partial pairing). Rate differences with stratified paired bootstrap CI and exact McNemar test on the discordant pairs.
"""
import glob, json, re, sys, collections, math
import numpy as np
from scipy import stats

B = 4000
R = np.random.default_rng(2024)
RUNS = 'results/s2/runs'
B_BUDGET = 0.10
LINES = []
def P(s=''):
    print(s); LINES.append(s)

def load(cond, arm):
    eps = {}; runs = []
    for f in sorted(glob.glob(f'{RUNS}/{cond}_{arm}_m[0-9]_j[0-9]*.json')):
        mm = re.search(r'_m(\d)_j(\d+)\.json', f); m = int(mm.group(1))
        d = json.load(open(f))
        for slot, j, o, st, tag in d['episodes']:
            if tag == 'post': eps[(m, slot, j)] = (o, st)
        steps = None
        sf = f.replace('.json', '.steps.npz')
        try: steps = dict(np.load(sf))
        except Exception: pass
        runs.append(dict(m=m, d=d, steps=steps, file=f))
    return eps, runs

def rate(eps, keys, o): return np.mean([eps[k][0] == o for k in keys])

def boot_ci(eps, B=B):
    keys = list(eps); strata = collections.defaultdict(list)
    for k in keys: strata[k[0]].append(k)
    arr = {m: (np.array([eps[k][0] == 'success' for k in v]), np.array([eps[k][0] == 'fall' for k in v]), np.array([eps[k][0] == 'timeout' for k in v]),
               np.array([eps[k][1] if eps[k][0] == 'success' else np.nan for k in v], float)) for m, v in strata.items()}
    n = len(keys); out = np.zeros((B, 4))
    for b in range(B):
        tot = np.zeros(3); st = []
        for m, v in arr.items():
            idx = R.integers(0, len(v[0]), len(v[0]))
            tot += [v[0][idx].sum(), v[1][idx].sum(), v[2][idx].sum()]; s = v[3][idx]; st.append(s[~np.isnan(s)])
        out[b, :3] = tot / n; out[b, 3] = np.median(np.concatenate(st)) if sum(len(x) for x in st) else np.nan
    return np.nanpercentile(out, [2.5, 97.5], axis=0)

def summarize(name, eps):
    n = len(eps)
    if n == 0: return None
    keys = list(eps)
    pt = [rate(eps, keys, o) for o in ('success', 'fall', 'timeout')]
    sg = [eps[k][1] for k in keys if eps[k][0] == 'success']
    md = float(np.median(sg)) if sg else float('nan')
    ci = boot_ci(eps)
    return dict(name=name, n=n, success=pt[0], fall=pt[1], timeout=pt[2], median=md, ci=ci.tolist())

def fmt(r):
    c = r['ci']
    return (f"{r['name']:34s} {r['n']:4d}  {r['success']:.3f} [{c[0][0]:.3f},{c[1][0]:.3f}]  {r['fall']:.3f} [{c[0][1]:.3f},{c[1][1]:.3f}]  "
            f"{r['timeout']:.3f} [{c[0][2]:.3f},{c[1][2]:.3f}]  {r['median']:.1f} [{c[0][3]:.0f},{c[1][3]:.0f}]")

def mcnemar(a, b):
    """exact two-sided McNemar on discordant pairs; a, b boolean arrays"""
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b)); n = n01 + n10
    if n == 0: return 1.0, n10, n01
    return float(stats.binomtest(min(n01, n10), n, 0.5).pvalue), n10, n01

def paired(eps_a, eps_b, label):
    """b minus a on matched keys"""
    keys = [k for k in eps_a if k in eps_b]
    if len(keys) < 5: return None
    out = dict(label=label, n=len(keys))
    strata = np.array([k[0] for k in keys])
    for o in ('success', 'fall', 'timeout'):
        xa = np.array([eps_a[k][0] == o for k in keys]); xb = np.array([eps_b[k][0] == o for k in keys])
        d = xb.astype(float) - xa.astype(float)
        bs = []
        idx_by = {m: np.where(strata == m)[0] for m in np.unique(strata)}
        for _ in range(B):
            tot = 0.0
            for m, ix in idx_by.items(): tot += d[R.choice(ix, len(ix))].sum()
            bs.append(tot / len(keys))
        p, n10, n01 = mcnemar(xa, xb)
        out[o] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p, n10, n01)
    return out

def fmt_paired(p):
    if p is None: return '  (insufficient matched episodes)'
    s = f"  {p['label']:44s} n={p['n']:4d}"
    for o in ('success', 'fall', 'timeout'):
        d, lo, hi, pv, n10, n01 = p[o]
        s += f" | {o[:4]} {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f} ({n10}/{n01})"
    return s

def step_stats(runs, alpha=None):
    """commit / escalate / defer rates and true-simulator diagnostics over post-phase steps"""
    out = collections.defaultdict(list)
    for r in runs:
        s = r['steps']
        if s is None: continue
        msk = s['tag'] == 'post'
        st = s['stage'][msk]
        pT = np.where(st == 2, s['pT2'][msk], s['pT1'][msk])
        frp = s['Frp'][msk]
        cid = (r['m'] * 1000 + s['slot'][msk]) * 1000 + s['jep'][msk]
        out['stage'].append(st); out['pT'].append(pT); out['pT1'].append(s['pT1'][msk]); out['frp'].append(frp); out['cid'].append(cid)
        out['alpha_t'].append(s['alpha_t'][msk]); out['m'].append(np.full(msk.sum(), r['m']))
    if not out: return None
    c = {k: np.concatenate(v) for k, v in out.items()}
    return c

def cluster_boot(x, cid, B=1000):
    """bootstrap of mean(x) over clusters (episodes)"""
    ids, inv = np.unique(cid, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(ids)); cnt = np.bincount(inv, minlength=len(ids)).astype(float)
    bs = []
    for _ in range(B):
        ix = R.integers(0, len(ids), len(ids)); bs.append(sums[ix].sum() / cnt[ix].sum())
    return np.percentile(bs, [2.5, 97.5])

def scc_report(arm, runs):
    c = step_stats(runs)
    if c is None: return None
    st = c['stage']; n = len(st)
    com = st >= 1
    fc = (com & (c['pT'] > B_BUDGET)).astype(float)
    ok = ~np.isnan(c['pT'])
    d = dict(n_steps=n, commit1=float(np.mean(st == 1)), commit2=float(np.mean(st == 2)), defer=float(np.mean(st == 0)),
             fcr=float(fc.mean()), fcr_ci=cluster_boot(fc, c['cid']).tolist(),
             mean_pT_commit=float(np.nanmean(c['pT'][com])) if com.any() else float('nan'), mean_pT_all=float(np.nanmean(c['pT1'])),
             mean_pT_defer=float(np.nanmean(c['pT1'][~com])) if (~com).any() else float('nan'),
             frp_commit=float(np.mean(c['frp'][com & (c['frp'] >= 0)])) if (com & (c['frp'] >= 0)).any() else float('nan'),
             risky_rate=float(np.mean(c['pT1'] > B_BUDGET)), recall=float(np.mean(~com[c['pT1'] > B_BUDGET])) if (c['pT1'] > B_BUDGET).any() else float('nan'),
             precision=float(np.mean(c['pT1'][~com] > B_BUDGET)) if (~com).any() else float('nan'),
             alpha_t_mean=float(np.mean(c['alpha_t'])), alpha_t_end=float(c['alpha_t'][-1]),
             rows=float(np.mean([r['d']['rows_per_env_step'] for r in runs])), sec=float(np.mean([r['d']['sec'] for r in runs])))
    return d

def fmt_scc(arm, d, alpha):
    return (f"  {arm:14s} steps {d['n_steps']:6d} commit s1 {d['commit1']:.3f} s2 {d['commit2']:.3f} defer {d['defer']:.3f} | "
            f"FCR (commit & pT>{B_BUDGET}) {d['fcr']:.4f} [{d['fcr_ci'][0]:.4f},{d['fcr_ci'][1]:.4f}] vs alpha {alpha} | mean pT: all {d['mean_pT_all']:.4f} commit {d['mean_pT_commit']:.4f} "
            f"defer {d['mean_pT_defer']:.4f} | gate: P(base plan risky pT>b) {d['risky_rate']:.3f}, risky steps deferred {d['recall']:.3f}, deferred steps risky {d['precision']:.3f} | replayed-fall rate of commits {d['frp_commit']:.4f} | alpha_t mean {d['alpha_t_mean']:.3f} end {d['alpha_t_end']:.3f} | rows/env-step {d['rows']:.0f}")

ARMS = [  # (arm key, label, alpha)
    ('base', 'base open loop N=32', None), ('cvar', 'base + CVaR mix 0.5', None), ('ctrl', 'base + failure-rate ctrl (0.05)', None),
    ('scc05', 'SCC alpha=0.05 (static)', 0.05), ('scc10', 'SCC alpha=0.10 (static)', 0.10),
    ('scc05aci', 'SCC alpha=0.05 + ACI', 0.05), ('scc10aci', 'SCC alpha=0.10 + ACI', 0.10),
    ('eq43', 'base open loop N=43 (equal compute)', None), ('rand05', 'random deferral p=0.22 (match SCC.05) N=32', None), ('rand10', 'random deferral p=0.15 (match SCC.10) N=32', None),
    ('pw10', 'SCC pointwise band alpha=0.10 N=32', 0.10), ('pw10b', 'SCC pointwise band alpha=0.10 N=32 (b=0.10)', 0.10),
    ('sel64', 'SCC selected-plan band alpha=0.10 N=64', 0.10), ('pw64', 'SCC pointwise band alpha=0.10 N=64', 0.10), ('base64', 'base open loop N=64', None),
    ('pi0', 'pi_0 only (v0=0.5)', None), ('pi0fast', 'pi_0 fast only (v0=1.0)', None)]
COND = {'in': 'IN DISTRIBUTION (wind 0.09)', 'sh': 'SHIFT (wind 0.09 -> 0.13 after 200 burn-in steps; post-shift episodes only; no retraining)'}
SUMMARY = {}
for cond in ('in', 'sh'):
    P('=' * 150); P(COND[cond]); P('=' * 150)
    P(f"{'arm':34s} {'eps':>4s}  {'success [95% CI]':20s}  {'fall [95% CI]':20s}  {'timeout [95% CI]':20s}  median steps (successes)")
    E, RU, SM = {}, {}, {}
    for arm, label, alpha in ARMS:
        eps, runs = load(cond, arm)
        if not eps: continue
        E[arm], RU[arm] = eps, runs
        r = summarize(label, eps); SM[arm] = r; P(fmt(r))
    SUMMARY[cond] = {a: v for a, v in SM.items()}
    P('')
    P('Compute (predictor rows per env step; base N=32, M=8, H=10, 3 iterations = 7680):')
    for arm, label, alpha in ARMS:
        if arm in RU and arm not in ('pi0', 'pi0fast'):
            P(f"  {label:44s} {np.mean([r['d']['rows_per_env_step'] for r in RU[arm]]):9.0f}   wall s/run {np.mean([r['d']['sec'] for r in RU[arm]]):6.0f}")
    P('')
    P('SCC gate diagnostics (steps of the post phase; FCR = P(commit and true open-loop H-step fall probability of the committed plan > b=%.2f), true simulator, diagnostic only):' % B_BUDGET)
    for arm, label, alpha in ARMS:
        if arm in RU and (arm.startswith('scc') or arm.startswith('pw') or arm.startswith('sel')):
            d = scc_report(arm, RU[arm])
            if d: P(fmt_scc(arm, d, alpha)); SUMMARY[cond][arm + '_gate'] = d
    P('')
    P('Paired comparisons (b minus a) on matched (model, slot, episode) pairs; cells: difference [95% CI] McNemar p (a-only successes... discordant n10/n01 = a-event-only / b-event-only)')
    for ref in ('base', 'cvar'):
        if ref not in E: continue
        for arm, label, alpha in ARMS:
            if arm == ref or arm not in E: continue
            P(fmt_paired(paired(E[ref], E[arm], f'{arm} minus {ref}')))
        P('')

    P('Key contrasts (b minus a), matched episodes:')
    for a_, b_ in (('rand05', 'scc05'), ('rand10', 'scc10'), ('pi0', 'scc05'), ('pi0', 'scc10'), ('eq43', 'scc10'), ('scc10', 'pw10'), ('scc05', 'scc05aci'), ('scc10', 'scc10aci'),
                   ('base', 'rand05'), ('base', 'rand10'), ('base', 'eq43'), ('pi0', 'pi0fast'), ('scc10', 'sel64'), ('pw64', 'sel64')):
        if a_ in E and b_ in E:
            P(fmt_paired(paired(E[a_], E[b_], f'{b_} minus {a_}')))
    P('')
P('=' * 150); P('CONTINUE / STOP CRITERIA (N3), post-shift episodes'); P('=' * 150)
SMsh = SUMMARY['sh']
P('(a) fall rate under shift with ACI within +-0.02 of the target (target = alpha of the arm; the fall rate floor of the fallback alone is the pi_0 row):')
for arm, alpha in (('scc05aci', 0.05), ('scc10aci', 0.10), ('scc05', 0.05), ('scc10', 0.10)):
    if arm in SMsh:
        r = SMsh[arm]; c = r['ci']
        P(f"  {arm:10s} fall {r['fall']:.3f} [{c[0][1]:.3f},{c[1][1]:.3f}] target {alpha:.2f}  |fall - target| = {abs(r['fall']-alpha):.3f}  -> {'WITHIN' if abs(r['fall']-alpha) <= 0.02 else 'NOT within'} +-0.02"
          f"   (interval overlaps [target-0.02, target+0.02]: {c[0][1] <= alpha + 0.02 and c[1][1] >= alpha - 0.02})")
for k in ('pi0', 'pi0fast'):
    if k in SMsh: P(f"  reference {k}: fall {SMsh[k]['fall']:.3f} [{SMsh[k]['ci'][0][1]:.3f},{SMsh[k]['ci'][1][1]:.3f}] (an always-defer policy cannot go below this)")
P('(b) success at equal (or lower) fall rate at least 0.03 above the CVaR arm (paired, post-shift; in distribution below):')
for cond_, E_ in (('sh', E), ):
    for arm in ('scc05aci', 'scc10aci', 'scc05', 'scc10'):
        if arm in E_ and 'cvar' in E_:
            p = paired(E_['cvar'], E_[arm], f'{arm} minus cvar'); d, lo, hi = p['success'][:3]; fd, flo, fhi = p['fall'][:3]
            P(f"  {arm:10s} success diff {d:+.3f} [{lo:+.3f},{hi:+.3f}]  fall diff {fd:+.3f} [{flo:+.3f},{fhi:+.3f}]  -> success gain >= 0.03: {d >= 0.03} (CI lower bound >= 0.03: {lo >= 0.03}); fall not higher: {fd <= 0.0 or flo <= 0 <= fhi}")
P('')
open('results/s2/summary.txt', 'w').write('\n'.join(LINES) + '\n')
json.dump(SUMMARY, open('results/s2/summary.json', 'w'), default=float)
