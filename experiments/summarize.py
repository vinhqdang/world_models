"""Aggregate suite results: mean +/- bootstrap 95% CI over (model seed x planning episodes), paired differences."""
import glob, json, os, sys, re
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else 'suite_fixed'
rng = np.random.default_rng(0)
rows = {}
for f in sorted(glob.glob(f'{root}/res/*.json')):
    m = re.match(r'(.+)_s(\d+)\.json', os.path.basename(f))
    d = json.load(open(f)); rows.setdefault(m.group(1), {})[int(m.group(2))] = d

def ci(x, n=4000):
    x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])

print(f'{"config":20s} {"runs":>4s} {"episodes":>8s} {"success":>8s} {"95% CI":>16s} {"fall":>7s} {"timeout":>8s} {"steps2goal":>10s}')
for name, seeds in rows.items():
    suc = [seeds[s]['success'] for s in sorted(seeds)]; fal = [seeds[s]['fall'] for s in sorted(seeds)]; to = [seeds[s]['timeout'] for s in sorted(seeds)]
    ep = sum(seeds[s]['episodes'] for s in seeds)
    lo, hi = ci(suc) if len(suc) > 1 else (np.nan, np.nan)
    ms = [seeds[s]['median_steps'] for s in seeds if seeds[s]['median_steps']]
    print(f'{name:20s} {len(seeds):4d} {ep:8d} {np.mean(suc):8.3f} [{lo:5.3f},{hi:5.3f}] {np.mean(fal):7.3f} {np.mean(to):8.3f} {np.mean(ms) if ms else float("nan"):10.1f}')
