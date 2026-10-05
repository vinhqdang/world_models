"""Pooled oracle summary from results/d8/oracle_<arm>_e<seed>.json -> results/d8/oracle_summary.txt.
Streaming-window length bias removed by keeping only episodes that started at or before total_steps-120 (column 'startcut')."""
import json, glob, re, collections
import numpy as np
from scipy import stats
R = np.random.default_rng(7); B = 2000
runs = collections.defaultdict(dict); meta = {}
for f in sorted(glob.glob('results/d8/oracle_*_e*.json')):
    m = re.match(r'results/d8/oracle_(.+)_e(\d+)\.json', f); arm, e = m.group(1), int(m.group(2))
    d = json.load(open(f)); cum = collections.defaultdict(int); r = {}
    for s, j, o, st in sorted(d['records'], key=lambda x: (x[0], x[1])):
        cum[s] += st; r[(s, j)] = (o, st, cum[s] - st)       # outcome, steps, start step
    runs[arm][e] = (r, d['args']['total_steps'], d['args']['ep_len']); meta[arm] = d
def eps(arm, cut=False):
    out = []
    for e, (r, T, L) in runs[arm].items():
        out += [(e, s, j, o, st) for (s, j), (o, st, s0) in r.items() if (not cut or s0 <= T - L)]
    return out
def rates(rows):
    n = len(rows); c = collections.Counter(o for *_, o, _ in rows)
    return n, c['success'] / n, c['fall'] / n, c['timeout'] / n, c['fall'] / max(1, c['fall'] + c['success'])
def bootci(rows, k):
    n = len(rows); o = np.array([r[3] for r in rows])
    v = []
    for _ in range(B):
        x = o[R.integers(0, n, n)]; s = (x == 'success').sum(); f = (x == 'fall').sum(); t = (x == 'timeout').sum()
        v.append([s / n, f / n, t / n, f / max(1, f + s)])
    return np.percentile(np.array(v)[:, k], [2.5, 97.5])
order = [a for a in ['ol32', 'ol36', 'fbs75', 'fbs150', 'ol32_lam05', 'fbs75_lam05'] if a in runs]
L = ['Oracle (true simulator), cliff_hi, E=100, 500 steps/run, eval seeds 100-104, H=10 iters=3 M=8; CIs = episode bootstrap (pooled over seeds)',
     'hazard = fall/(fall+success) = conditional fall rate among episodes that were decided (not timed out)', '']
for cut in (False, True):
    L.append('ALL completed episodes' if not cut else 'START-CUT subset (length-bias free)')
    L.append(f"{'arm':12s} {'seeds':>5s} {'eps':>6s} {'success':>20s} {'fall':>20s} {'timeout':>20s} {'hazard':>20s} part-steps")
    for a in order:
        rows = eps(a, cut); n, s, f, t, h = rates(rows)
        ci = [bootci(rows, k) for k in range(4)]
        L.append(f"{a:12s} {len(runs[a]):5d} {n:6d} " + ' '.join(f"{v:.3f} [{c[0]:.3f},{c[1]:.3f}]".rjust(20) for v, c in zip((s, f, t, h), ci)) + f" {meta[a]['particle_steps_per_env_step']}")
    L.append('')
L.append('Unpaired two-proportion tests (arm vs baseline), all completed episodes; paired = same (seed, slot, episode idx) start/goal, McNemar')
for base in ['ol32', 'ol36']:
    for a in order:
        if a == base or base not in runs or (base == 'ol36' and a == 'ol32'): continue
        ra, rb = eps(base), eps(a); out = []
        for nm in ('success', 'fall', 'timeout'):
            xa = np.array([r[3] == nm for r in ra]); xb = np.array([r[3] == nm for r in rb])
            p0 = (xa.sum() + xb.sum()) / (len(xa) + len(xb)); se = np.sqrt(p0 * (1 - p0) * (1 / len(xa) + 1 / len(xb)))
            out.append(f"{nm} {xb.mean() - xa.mean():+.3f} (p={2 * stats.norm.sf(abs((xb.mean() - xa.mean()) / se)):.1e})")
        da = {(r[0], r[1], r[2]): r[3] for r in ra}; db = {(r[0], r[1], r[2]): r[3] for r in rb}
        ks = [k for k in da if k in db]; pm = []
        for nm in ('success', 'fall'):
            n01 = sum(1 for k in ks if db[k] == nm and da[k] != nm); n10 = sum(1 for k in ks if da[k] == nm and db[k] != nm)
            pm.append(f"{nm} McNemar p={stats.binomtest(n01, n01 + n10, .5).pvalue:.1e} ({n01}/{n10})")
        # hazard difference test (two-proportion on fall vs success among decided)
        fa, sa_ = sum(r[3] == 'fall' for r in ra), sum(r[3] == 'success' for r in ra); fb_, sb_ = sum(r[3] == 'fall' for r in rb), sum(r[3] == 'success' for r in rb)
        ha, hb = fa / (fa + sa_), fb_ / (fb_ + sb_); p0 = (fa + fb_) / (fa + sa_ + fb_ + sb_); se = np.sqrt(p0 * (1 - p0) * (1 / (fa + sa_) + 1 / (fb_ + sb_)))
        L.append(f"{a} vs {base}: " + '; '.join(out) + f"; hazard {hb - ha:+.3f} (p={2 * stats.norm.sf(abs((hb - ha) / se)):.1e}); paired n={len(ks)}: " + '; '.join(pm))
txt = '\n'.join(L); print(txt); open('results/d8/oracle_summary.txt', 'w').write(txt + '\n')
