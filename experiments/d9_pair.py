"""Pairwise unpaired episode-bootstrap differences between pooled arms (fbstep_crn3 vs each other arm)."""
import glob, json
import numpy as np
rng = np.random.default_rng(1)
def eps(arm): return np.array([e[0] for f in sorted(glob.glob(f'results/d9/{arm}_m[0-9].json')) for e in json.load(open(f))['episode_list']])
c = eps('fbstep_crn3'); out = []
for o in ('ol_indep', 'ol_crn3', 'fbstep_indep'):
    b = eps(o); row = f'fbstep_crn3 - {o}: '
    for k in ('success', 'fall', 'timeout'):
        d = [(c[rng.integers(0, len(c), len(c))] == k).mean() - (b[rng.integers(0, len(b), len(b))] == k).mean() for _ in range(5000)]
        row += f'd{k}={(c==k).mean()-(b==k).mean():+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  '
    out.append(row); print(row)
open('results/d9/pairwise.txt', 'w').write('\n'.join(out) + '\n')
