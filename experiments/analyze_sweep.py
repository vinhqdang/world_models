"""Analyse a budget sweep produced by experiments/vm/sweep1.sh.

For every (N, seed) run we use only *full-chunk* outcomes (the executed plan ran its whole
length, i.e. the episode was not terminated mid-chunk), because only then is the model-predicted
terminal cost comparable with the realised terminal cost.
"""
import glob
import json
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'results/lewm/sweep1'
rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


runs = {}
for f in sorted(glob.glob(os.path.join(root, '*.json'))):
    m = re.search(r'N(\d+)_S(\d+)_seed(\d+)', f)
    if not m:
        continue
    N, S, seed = map(int, m.groups())
    d = json.load(open(f))
    runs.setdefault((N, S), []).append(d)

print(f'{"N":>6} {"S":>3} {"runs":>4} {"succ%":>7} | {"n_full":>6} {"c_hat":>8} {"y_real":>8} {"gap y-c":>8} '
      f'{"prog_gold":>9} {"prog/p0":>8} | {"corr(c,y)":>9}')
summary = {}
for (N, S), ds in sorted(runs.items()):
    succ = [x['extra']['success_rate'] for x in ds]
    rows = [r for d in ds for r in d['rows'] if 'y' in r and not r.get('partial', False)]
    if not rows:
        print(f'{N:>6} {S:>3} {len(ds):>4} {np.mean(succ):7.1f} | no full-chunk rows')
        continue
    c = np.array([r['c_hat'] for r in rows])
    y = np.array([r['y'] for r in rows])
    p0 = np.array([r['p_start'] for r in rows])
    p1 = np.array([r['p_end'] for r in rows])
    prog = p0 - p1
    rel = prog / np.maximum(p0, 1e-6)
    corr = np.corrcoef(c, y)[0, 1] if len(rows) > 3 else np.nan
    summary[(N, S)] = dict(succ=succ, c=c, y=y, prog=prog, rel=rel)
    lo, hi = boot_ci(prog)
    print(f'{N:>6} {S:>3} {len(ds):>4} {np.mean(succ):7.1f} | {len(rows):>6} {c.mean():8.1f} {y.mean():8.1f} '
          f'{(y - c).mean():8.1f} {prog.mean():6.2f}[{lo:5.2f},{hi:5.2f}] {rel.mean():8.3f} | {corr:9.3f}')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Ns = sorted({k[0] for k in summary})
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, (key, title) in zip(ax, [('succ', 'success rate (%)'), ('c', 'predicted terminal cost'),
                                    ('y', 'realised terminal cost'), ('prog', 'true progress (proprio)')]):
        mu = [np.mean(summary[(n, 30)][key]) for n in Ns if (n, 30) in summary]
        a.plot([n for n in Ns if (n, 30) in summary], mu, 'o-')
        a.set_xscale('log'); a.set_xlabel('candidates N'); a.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(root, 'sweep1.png'), dpi=110)
    print('saved', os.path.join(root, 'sweep1.png'))
except Exception as e:
    print('plot skipped:', e)
