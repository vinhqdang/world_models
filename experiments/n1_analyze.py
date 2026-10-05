"""Summaries for N1: pooled outcome rates with stratified bootstrap CIs, paired (McNemar, matched by run/slot/episode index)
and unpaired differences.  usage: n1_analyze.py DIR ref_arm arm1,arm2,...   (DIR = results/n1/learned or results/n1/oracle)"""
import glob, json, os, sys, re
import numpy as np
from scipy.stats import binomtest, norm

d, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
OUT = ('success', 'fall', 'timeout')
rng = np.random.default_rng(0)


def load(arm):
    runs = {}
    for f in sorted(glob.glob(f'{d}/{arm}_*.json')):
        if not re.match(rf'^{arm}_(m\d_)?e\d+\.json$', os.path.basename(f)):
            continue
        r = json.load(open(f)); runs[os.path.basename(f)[len(arm) + 1:-5]] = r
    return runs


data = {a: load(a) for a in [ref] + [x for x in arms if x != ref]}


def pooled(runs):
    cnt = {o: sum(r['counts'][o] for r in runs.values()) for o in OUT}
    return cnt, sum(cnt.values())


def boot(runs, B=4000):
    """stratified-by-run bootstrap of the three rates"""
    recs = {k: [e[2] for e in r['records']] for k, r in runs.items()}
    out = {o: [] for o in OUT}
    for _ in range(B):
        tot = {o: 0 for o in OUT}; n = 0
        for k, v in recs.items():
            v = np.array(v); s = v[rng.integers(0, len(v), len(v))]
            for o in OUT: tot[o] += int((s == o).sum())
            n += len(v)
        for o in OUT: out[o].append(tot[o] / n)
    return {o: np.percentile(out[o], [2.5, 97.5]) for o in OUT}


def paired(runs_a, runs_b, o):
    """matched by (run key, slot, episode index): b = arm, a = ref;  returns n, b10 (arm has o, ref not), b01, diff, McNemar p"""
    b10 = b01 = n = 0
    for k in runs_a:
        if k not in runs_b: continue
        A = {(e[0], e[1]): e[2] for e in runs_a[k]['records']}
        B = {(e[0], e[1]): e[2] for e in runs_b[k]['records']}
        for key in A.keys() & B.keys():
            n += 1; xa, xb = A[key] == o, B[key] == o
            if xb and not xa: b10 += 1
            if xa and not xb: b01 += 1
    p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 > 0 else 1.0
    return n, b10, b01, (b10 - b01) / max(1, n), p


def two_prop(c1, n1, c2, n2):
    p = (c1 + c2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * (1 - norm.cdf(abs(c1 / n1 - c2 / n2) / se)) if se > 0 else 1.0


def diff_boot(runs_a, runs_b, o, B=4000):
    """unpaired stratified bootstrap CI of rate(b) - rate(a)"""
    ra = {k: np.array([e[2] == o for e in r['records']], float) for k, r in runs_a.items()}
    rb = {k: np.array([e[2] == o for e in r['records']], float) for k, r in runs_b.items()}
    ds = []
    for _ in range(B):
        sa = sum(v[rng.integers(0, len(v), len(v))].sum() for v in ra.values()) / sum(len(v) for v in ra.values())
        sb = sum(v[rng.integers(0, len(v), len(v))].sum() for v in rb.values()) / sum(len(v) for v in rb.values())
        ds.append(sb - sa)
    return np.percentile(ds, [2.5, 97.5])


print(f'### {d}   (reference arm: {ref})')
print('| arm | runs | episodes | success [95% CI] | fall [95% CI] | timeout [95% CI] | median steps to goal |')
print('|---|---|---|---|---|---|---|')
for a, runs in data.items():
    if not runs: continue
    cnt, n = pooled(runs); ci = boot(runs)
    st = [e[3] for r in runs.values() for e in r['records'] if e[2] == 'success']
    cells = ' | '.join(f'{cnt[o]/n:.3f} [{ci[o][0]:.3f}, {ci[o][1]:.3f}]' for o in OUT)
    print(f'| {a} | {len(runs)} | {n} | {cells} | {np.median(st):.1f} |')
print()
print('Differences (arm minus reference): paired = matched (run, slot, episode index), McNemar exact p; unpaired = stratified bootstrap CI')
print('| arm | outcome | n matched | discordant (arm-only / ref-only) | paired diff | McNemar p | unpaired diff [95% CI] | z-test p |')
print('|---|---|---|---|---|---|---|---|')
cr, nr = pooled(data[ref])
for a, runs in data.items():
    if a == ref or not runs: continue
    ca, na = pooled(runs)
    for o in OUT:
        n_, b10, b01, dd, p = paired(data[ref], runs, o)
        lo, hi = diff_boot(data[ref], runs, o)
        print(f'| {a} | {o} | {n_} | {b10}/{b01} | {dd:+.3f} | {p:.3g} | {ca[o]/na-cr[o]/nr:+.3f} [{lo:+.3f}, {hi:+.3f}] | {two_prop(ca[o], na, cr[o], nr):.3g} |')
print()
print('per run (success / fall / timeout, episodes):')
for a, runs in data.items():
    for k, r in runs.items():
        print(f'  {a:12s} {k:10s} {r["success"]:.3f} / {r["fall"]:.3f} / {r["timeout"]:.3f}  ({r["episodes"]})' +
              (f'  optB-A={r["optimism_B_minus_A_mean"]:.3f} mean-plan-chosen={r["frac_mean_plan_chosen"]:.3f}' if r.get('optimism_B_minus_A_mean') is not None else ''))
