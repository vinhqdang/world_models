"""Sample-efficiency table: per M and arm, pooled over the listed model seeds (bootstrap 95% CI over episodes)."""
import glob, json, os, re
import numpy as np
rng = np.random.default_rng(0)
def get(arm, M, seeds):
    suf = '' if M == 8 else f'_M{M}'
    eps, n = [], []
    for s in seeds:
        f = f'results/d9/{arm}{suf}_m{s}.json'
        if os.path.exists(f):
            r = json.load(open(f)); eps += [tuple(e) for e in r['episode_list']]; n.append(s)
    return eps, n
def row(eps):
    a = np.array([e[0] for e in eps]); B = 3000
    idx = rng.integers(0, len(a), (B, len(a)))
    out = []
    for k in ('success', 'fall', 'timeout'):
        m = (a == k).mean(); bs = np.array([(a[i] == k).mean() for i in idx])
        out.append(f"{m:.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
    return out
lines = []
for seeds in ([0], [0, 1]):
    lines.append(f'--- model seeds {seeds} (eval seed 211, E=16 N=32 H=10 iters=3, 450 steps) ---')
    lines.append(f"{'arm':14s} {'M':>3s} {'eps':>4s}  {'success [95% CI]':22s} {'fall [95% CI]':22s} {'timeout [95% CI]':22s}")
    for arm in ['ol_indep', 'ol_crn3', 'fbstep_indep', 'fbstep_crn3']:
        for M in (4, 8, 16):
            eps, n = get(arm, M, seeds)
            if len(n) != len(seeds): continue
            lines.append(f"{arm:14s} {M:3d} {len(eps):4d}  " + '  '.join(f'{x:22s}' for x in row(eps)))
print('\n'.join(lines)); open('results/d9/sweep_summary.txt', 'w').write('\n'.join(lines) + '\n')
