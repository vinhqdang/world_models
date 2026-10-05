"""Summarise results/d9: pooled over model seeds, bootstrap 95% CI over episodes, per-model-seed numbers.
usage: python experiments/d9_summ.py [suffix-regex-prefix ...]   (default: the four main arms)"""
import glob, json, re, sys, collections
import numpy as np
rng = np.random.default_rng(0)
arms = sys.argv[1:] or ['ol_indep', 'ol_crn3', 'fbstep_indep', 'fbstep_crn3']

def load(arm):
    rs = []
    for f in sorted(glob.glob(f'results/d9/{arm}_m[0-9].json')):
        rs.append((int(re.search(r'_m(\d)\.json', f).group(1)), json.load(open(f))))
    return rs

def metrics(eps):
    n = len(eps); o = [e[0] for e in eps]
    st = [e[1] for e in eps if e[0] == 'success']
    return np.array([o.count('success') / n, o.count('fall') / n, o.count('timeout') / n, np.median(st) if st else np.nan])

def boot(eps, B=4000):
    n = len(eps); idx = rng.integers(0, n, (B, n))
    arr = np.array([e[0] for e in eps]); st = np.array([e[1] for e in eps])
    out = np.zeros((B, 4))
    for b in range(B):
        a, s = arr[idx[b]], st[idx[b]]
        ok = s[a == 'success']
        out[b] = [(a == 'success').mean(), (a == 'fall').mean(), (a == 'timeout').mean(), np.median(ok) if len(ok) else np.nan]
    return np.nanpercentile(out, [2.5, 97.5], axis=0)

lines = []
P = lambda s: (print(s), lines.append(s))
P(f"{'arm':14s} {'eps':>4s}  {'success [95% CI]':22s} {'fall [95% CI]':22s} {'timeout [95% CI]':22s} {'med steps [CI]':18s} sec/run  rows/step")
pooled = {}
for arm in arms:
    rs = load(arm)
    if not rs: continue
    eps = [tuple(e) for _, r in rs for e in r['episode_list']]
    pooled[arm] = eps
    m = metrics(eps); ci = boot(eps)
    f = lambda i: f"{m[i]:.3f} [{ci[0][i]:.3f},{ci[1][i]:.3f}]"
    P(f"{arm:14s} {len(eps):4d}  {f(0):22s} {f(1):22s} {f(2):22s} {m[3]:.1f} [{ci[0][3]:.0f},{ci[1][3]:.0f}]   {np.mean([r['sec'] for _, r in rs]):.0f}  {np.mean([r['pred_rows_per_step'] for _, r in rs]):.0f}   (model seeds {[s for s, _ in rs]})")
P('')
P('per model seed (success / fall / timeout / median steps, episodes):')
for arm in arms:
    for s, r in load(arm):
        P(f"  {arm:14s} m{s}: eps={r['episodes']:3d} success={r['success']:.3f} fall={r['fall']:.3f} timeout={r['timeout']:.3f} med={r['median_steps']}  (counts {r['counts']})")
# paired bootstrap differences vs baseline (episodes are not paired 1:1, so use unpaired bootstrap)
if 'ol_indep' in pooled:
    P('')
    P('difference vs ol_indep (unpaired episode bootstrap 95% CI):')
    base = pooled['ol_indep']
    for arm, eps in pooled.items():
        if arm == 'ol_indep': continue
        B = 4000; out = np.zeros((B, 3))
        a0 = np.array([e[0] for e in base]); a1 = np.array([e[0] for e in eps])
        for b in range(B):
            x = a0[rng.integers(0, len(a0), len(a0))]; y = a1[rng.integers(0, len(a1), len(a1))]
            out[b] = [(y == k).mean() - (x == k).mean() for k in ('success', 'fall', 'timeout')]
        d = [(a1 == k).mean() - (a0 == k).mean() for k in ('success', 'fall', 'timeout')]
        ci = np.percentile(out, [2.5, 97.5], axis=0)
        P(f"  {arm:14s} dsucc={d[0]:+.3f} [{ci[0][0]:+.3f},{ci[1][0]:+.3f}]  dfall={d[1]:+.3f} [{ci[0][1]:+.3f},{ci[1][1]:+.3f}]  dtimeout={d[2]:+.3f} [{ci[0][2]:+.3f},{ci[1][2]:+.3f}]")
open('results/d9/summary.txt' if not sys.argv[1:] else '/dev/null', 'w').write('\n'.join(lines) + '\n')
