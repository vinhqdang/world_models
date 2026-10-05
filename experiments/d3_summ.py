import glob, json, sys, re, collections, numpy as np
pre = sys.argv[1]
g = collections.defaultdict(list)
for f in sorted(glob.glob(f'results/d3/{pre}*_s[0-9].json')):
    g[re.sub(r'_s\d\.json$', '', f.split('/')[-1])].append(json.load(open(f)))
print(f"{'config':28s} runs   eps  success   fall  timeout  medsteps  sec/run")
for k, rs in g.items():
    ep = sum(r['episodes'] for r in rs)
    w = lambda m: sum(r[m] * r['episodes'] for r in rs) / ep
    per = [r['success'] for r in rs]
    ms = np.mean([r['median_steps'] for r in rs if r['median_steps']])
    print(f"{k:28s} {len(rs):3d} {ep:5d}  {w('success'):.3f}  {w('fall'):.3f}  {w('timeout'):.3f}   {ms:5.1f}   {np.mean([r['sec'] for r in rs]):.0f}   seedSD(succ)={np.std(per,ddof=1) if len(per)>1 else 0:.3f}")
