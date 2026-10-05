"""N2 analysis: (a) ranking-benchmark tables with scenario-level bootstrap CIs and paired differences; (b) closed-loop tables."""
import glob, json, sys
import numpy as np
from scipy.stats import binomtest

rng = np.random.default_rng(0)
B = 4000


def ci(x, stat=np.mean):
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x)))
    v = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def rank_tables(path='results/n2/rank_main.json', out='results/n2/rank_summary.md'):
    R = json.load(open(path))['results']
    for extra, nm in [('results/n2/rank_learned.json', 'learned'), ('results/n2/rank_moment.json', 'moment')]:
        try:
            RL = json.load(open(extra))['results']
            for k in R:
                if k in RL and nm in RL[k]: R[k][nm] = RL[k][nm]
        except FileNotFoundError:
            pass
    regimes = sorted({k.split('_', 1)[1] for k in R})
    schemes = [k for k in next(iter(R.values())) if not k.startswith('_')]
    lines = []
    metrics = [('spearman', 'Spearman'), ('rmse', 'rmse(centred)'), ('regret', 'regret'), ('bad0.3', 'P(regret>0.3)'), ('sel_fail', 'sel. fail prob'), ('elite_cost', 'elite ref cost')]
    for regime in regimes:
        keys = [k for k in R if k.endswith('_' + regime)]
        lines.append(f'\n### Regime `{regime}` ({len(keys)} models x {len(R[keys[0]][schemes[0]]["per_scen"]["spearman"])} scenarios)\n')
        lines.append('| scheme | evals/cand | det. | ' + ' | '.join(m[1] for m in metrics) + ' |')
        lines.append('|---|---|---|' + '---|' * len(metrics))
        for s in schemes:
            if not all(s in R[k] for k in keys): continue
            row = [s, f'{np.mean([R[k][s]["evals_per_cand"] for k in keys]):.0f}', 'y' if R[keys[0]][s]['deterministic'] else 'n']
            for m, _ in metrics:
                x = np.concatenate([R[k][s]['per_scen'][m] for k in keys])
                mu, lo, hi = ci(x)
                row.append(f'{mu:.3f} [{lo:.3f}, {hi:.3f}]')
            lines.append('| ' + ' | '.join(row) + ' |')
        for base in ['indep', 'crn_anti']:
            lines.append(f'\nPaired differences vs `{base}` (scenario bootstrap; negative is better for rmse/regret/P(regret)/fail/cost, positive for Spearman):\n')
            lines.append('| scheme | ' + ' | '.join(m[1] for m in metrics) + ' |')
            lines.append('|---|' + '---|' * len(metrics))
            for s in schemes:
                if s == base or not all(s in R[k] for k in keys): continue
                row = [s]
                for m, _ in metrics:
                    d = np.concatenate([np.array(R[k][s]['per_scen'][m]) - np.array(R[k][base]['per_scen'][m]) for k in keys])
                    mu, lo, hi = ci(d)
                    row.append(f'{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]')
                lines.append('| ' + ' | '.join(row) + ' |')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def cemeval(path='results/n2/cemeval.json', out='results/n2/cemeval_summary.md'):
    R = json.load(open(path))['results']
    schemes = list(next(iter(R.values())).keys())
    lines = ['| scheme | ref cost (mean over scenarios) | ref failure prob | paired cost diff vs crn_anti | paired fail diff vs crn_anti |', '|---|---|---|---|---|']
    for s_ in schemes:
        c = np.concatenate([R[m][s_]['cost'] for m in R]); f = np.concatenate([R[m][s_]['fail'] for m in R])
        cb = np.concatenate([R[m]['crn_anti']['cost'] for m in R]); fb = np.concatenate([R[m]['crn_anti']['fail'] for m in R])
        a, la, ha = ci(c); b, lb, hb = ci(f); dc = ci(c - cb); df = ci(f - fb)
        lines.append(f'| {s_} | {a:.3f} [{la:.3f}, {ha:.3f}] | {b:.3f} [{lb:.3f}, {hb:.3f}] | {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}] | {df[0]:+.3f} [{df[1]:+.3f}, {df[2]:+.3f}] |')
    lines.append('\nPer model (mean ref cost):\n')
    lines.append('| model | ' + ' | '.join(schemes) + ' |'); lines.append('|---|' + '---|' * len(schemes))
    for m in R:
        lines.append(f'| {m} | ' + ' | '.join(f'{np.mean(R[m][s_]["cost"]):.3f}' for s_ in schemes) + ' |')
    open(out, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))


def closed_loop(out='results/n2/cl_summary.md'):
    files = sorted(glob.glob('results/n2/cl_*_m*_e*.json'))
    arms = {}
    for f in files:
        d = json.load(open(f)); a = d['args']
        arms.setdefault(a['scheme'], {})[(int(a['ckpt'].split('_s')[1][0]), a['seed'])] = d['records']
    lines = ['| arm | runs | episodes | success | fall | timeout |', '|---|---|---|---|---|---|']
    for s, runs in arms.items():
        recs = [(k, r) for k, rr in runs.items() for r in rr]
        n = len(recs)
        def bs(o):
            ind = np.array([1.0 if r[2] == o else 0.0 for _, r in recs]); run_id = np.array([hash(k) for k, _ in recs])
            ks = list(runs.keys())
            per = [np.array([1.0 if r[2] == o else 0.0 for r in runs[k]]) for k in ks]
            vals = []
            for _ in range(B):
                num = den = 0
                for j in rng.integers(0, len(ks), len(ks)):
                    pass
                num = den = 0
                for p_ in per:
                    x = p_[rng.integers(0, len(p_), len(p_))]; num += x.sum(); den += len(x)
                vals.append(num / den)
            return np.mean(ind), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
        cells = [f'{m:.3f} [{lo:.3f}, {hi:.3f}]' for m, lo, hi in (bs(o) for o in ('success', 'fall', 'timeout'))]
        lines.append(f'| {s} | {len(runs)} | {n} | ' + ' | '.join(cells) + ' |')
    # paired vs baselines (matched model, seed, slot, episode index)
    for base in ['indep', 'crn_chain3']:
        if base not in arms: continue
        lines.append(f'\nPaired vs `{base}` (matched (model, seed, slot, episode-index) episodes; McNemar exact on success, and on fall):\n')
        lines.append('| arm | n matched | success diff | McNemar p (succ) | fall diff | McNemar p (fall) | timeout diff |')
        lines.append('|---|---|---|---|---|---|---|')
        for s, runs in arms.items():
            if s == base: continue
            tab = {o: [0, 0, 0, 0] for o in ('success', 'fall', 'timeout')}   # both, only arm, only base, neither
            n = 0
            for k, rr in runs.items():
                if k not in arms[base]: continue
                bm = {(r[0], r[1]): r[2] for r in arms[base][k]}
                for r in rr:
                    key = (r[0], r[1])
                    if key not in bm: continue
                    n += 1
                    for o in tab:
                        a_, b_ = r[2] == o, bm[key] == o
                        tab[o][0 if a_ and b_ else 1 if a_ else 2 if b_ else 3] += 1
            if n == 0: continue
            def diff(o):
                t = tab[o]; return (t[1] - t[2]) / n
            def p(o):
                t = tab[o]; m = t[1] + t[2]
                return binomtest(t[1], m, 0.5).pvalue if m > 0 else 1.0
            lines.append(f'| {s} | {n} | {diff("success"):+.3f} | {p("success"):.3f} | {diff("fall"):+.3f} | {p("fall"):.3f} | {diff("timeout"):+.3f} |')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if 'rank' in sys.argv: rank_tables()
    if 'cl' in sys.argv: closed_loop()
    if 'cem' in sys.argv: cemeval()
