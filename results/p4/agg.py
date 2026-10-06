import json, glob, numpy as np
# Pilot A
fs = sorted(glob.glob('pilot_a_s*.json')) 
res = [json.load(open(f)) for f in fs]
print('Pilot A seeds:', len(res))
print('oracle (source sensor, frozen P) succ', np.round([r['oracle_srcsensor'][0] for r in res], 2), 'mean', round(np.mean([r['oracle_srcsensor'][0] for r in res]), 3))
for n in ['250', '1000']:
    print('N =', n, 'target transitions; seeds', sum(1 for r in res if n in r))
    for m in res[0][n]:
        rr = [r for r in res if n in r]; su = [r[n][m]['succ'] for r in rr]; rc = [r[n][m].get('recov', np.nan) for r in rr]
        print(f'  {m:12s} succ mean {np.mean(su):.2f}  min {np.min(su):.2f} max {np.max(su):.2f} | per-seed {np.round(su,2)} | recov median {np.nanmedian(rc):.3f}  frac(recov<0.1) {np.mean(np.array(rc)<0.1) if m!="retrain" else float("nan"):.2f}')
# Pilot B
import os
out = {}
for f in sorted(glob.glob('pilot_b_s*_*.json')) + sorted(glob.glob('pilot_b_s*_random_local.json')):
    s = int(f.split('_s')[1].split('_')[0]); d = json.load(open(f))
    for k, v in d.items(): out.setdefault(k, {})[s] = [c[1] for c in v]
print('Pilot B (success, 80 tasks, true-model oracle 0.74, no-wall model 0.11); rounds x [300,504,708,912,1116,1320] samples')
for k, v in out.items():
    A = np.array([v[s] for s in sorted(v)]); print(f'  {k:12s} seeds {sorted(v)} final mean {A[:,-1].mean():.3f} (per-seed {np.round(A[:,-1],3)})  mean over rounds 2-5 {A[:,2:].mean():.3f}')
