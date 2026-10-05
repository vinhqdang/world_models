import json, glob, collections, numpy as np
rows = collections.defaultdict(list)
for f in sorted(glob.glob('results/d6/cl_*_s[0-9].json')):
    n = f.split('/')[-1][3:-8]; rows[n].append((int(f[-6]), json.load(open(f))))
for f in sorted(glob.glob('suite_hi/res/es_exp_s[0-9].json')):
    rows['BASELINE_indep(es_exp)'].append((int(f[-6]), json.load(open(f))))
out = []
for n, L in rows.items():
    for sub, name in ((None, n),) :
        pass
    seeds = sorted(s for s, _ in L)
    ep = sum(d['episodes'] for _, d in L)
    w = lambda k: sum(d[k] * d['episodes'] for _, d in L) / ep
    line = f"{n:28s} seeds={seeds} eps={ep:4d} success={w('success'):.3f} fall={w('fall'):.3f} timeout={w('timeout'):.3f}"
    # same-seed baseline comparison on seeds 0-1
    out.append(line)
    for s, d in sorted(L):
        out.append(f"    seed {s}: eps={d['episodes']} success={d['success']:.3f} fall={d['fall']:.3f} timeout={d['timeout']:.3f}")
txt = '\n'.join(out); print(txt); open('results/d6/summary.txt', 'w').write(txt + '\n')
