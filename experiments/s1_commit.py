"""S1 Phase A extra: commit/defer rule with a controlled false-commit rate (conformal risk control on the selected plan).

python experiments/s1_commit.py --scheme indep --bounds 0.1,0.2
A plan is 'bad' if its true pit-fall probability (400 true particles, sets A and B averaged) exceeds b.
Rule: commit iff J8(selected plan) <= lambda.  lambda_hat = largest lambda with  n/(n+1) * Rhat(lambda) + 1/(n+1) <= alpha,
Rhat(lambda) = mean over calibration states of 1[commit and bad]  (loss bounded by 1, nonincreasing as lambda decreases).
  SCC        : calibrated on the selected plans of the calibration states (same pool size N as at test time)
  pointwise  : calibrated on random pool members (all (state, member) pairs), then applied to the selected plan
False-commit rate (FCR) = P(commit and bad) on test states, averaged over reps and splits; Wilson 95% interval uses n_test.
Also reported: commit rate and P(bad | commit).
"""
import argparse, json, math
import numpy as np
from scipy.stats import beta as _beta

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', default='indep'); ap.add_argument('--pop', default='f32')
ap.add_argument('--bounds', default='0.1,0.2'); ap.add_argument('--alphas', default='0.05,0.1')
ap.add_argument('--reps', type=int, default=20); ap.add_argument('--splits', type=int, default=5)
ap.add_argument('--ks', default='0,1,2'); ap.add_argument('--Ns', default='2,4,8,16,32')
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(',')]
D = {k: np.load(f'results/s1/pops_es_s{k}_{args.scheme}.npz') for k in KS}
POP = args.pop


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def crc_lambda(score, bad, alpha):
    """largest lambda in {-inf} U scores with n/(n+1) mean(1[score<=lam & bad]) + 1/(n+1) <= alpha; commit iff score <= lambda."""
    n = len(score)
    o = np.argsort(score); s, b = score[o], bad[o].astype(float)
    cum = np.cumsum(b) / n                                    # Rhat at lambda = s[i] (ties: take the last of a tie group)
    last = np.r_[s[1:] != s[:-1], True]
    ok = (n / (n + 1)) * cum + 1.0 / (n + 1) <= alpha
    ok &= last
    return s[np.where(ok)[0].max()] if ok.any() else -np.inf


def ltt_lambda(score, bad, alpha, delta=0.1, step=5):
    """Conditional control P(bad | commit) <= alpha w.p. >= 1-delta (Learn-then-Test, fixed-sequence over nested commit sets,
    Clopper-Pearson test): commit the k states with the lowest scores for increasing k until the test fails."""
    o = np.argsort(score); s, b = score[o], bad[o].astype(int); lam = -np.inf
    cb = np.cumsum(b)
    for k in range(step, len(s) + 1, step):
        up = _beta.ppf(1 - delta, cb[k - 1] + 1, k - cb[k - 1])
        if up > alpha: break
        lam = s[k - 1]
    return lam


out = {'scheme': args.scheme, 'pop': POP, 'rows': []}
for b in [float(x) for x in args.bounds.split(',')]:
    for alpha in [float(x) for x in args.alphas.split(',')]:
        for N in [int(x) for x in args.Ns.split(',')]:
            rng = np.random.default_rng(5000 + N)
            m = {k: [] for k in ['fcr_scc', 'cr_scc', 'cond_scc', 'fcr_pt', 'cr_pt', 'cond_pt', 'p_bad', 'lam_scc', 'lam_pt', 'fcr_ltt', 'cr_ltt', 'cond_ltt']}
            for rep in range(args.reps):
                per = []
                for k in KS:
                    d = D[k]; J8 = d[f'{POP}_J8']; pf = 0.5 * (d[f'{POP}_pfA'] + d[f'{POP}_pfB'])
                    n, N0 = J8.shape
                    idx = np.argsort(rng.random((n, N0)), 1)[:, :N]
                    j8 = np.take_along_axis(J8, idx, 1); p_ = np.take_along_axis(pf, idx, 1)
                    sel = j8.argmin(1)[:, None]
                    per.append(dict(j8=j8, bad=p_ > b, sj=np.take_along_axis(j8, sel, 1)[:, 0], sbad=np.take_along_axis(p_, sel, 1)[:, 0] > b))
                for s in range(args.splits):
                    cal = dict(j=[], b=[], mj=[], mb=[]); tst = dict(j=[], b=[])
                    for p, k in zip(per, KS):
                        perm = np.random.default_rng([s, k, 77]).permutation(len(p['sj'])); c, t = perm[:len(perm) // 2], perm[len(perm) // 2:]
                        cal['j'].append(p['sj'][c]); cal['b'].append(p['sbad'][c]); cal['mj'].append(p['j8'][c].ravel()); cal['mb'].append(p['bad'][c].ravel())
                        tst['j'].append(p['sj'][t]); tst['b'].append(p['sbad'][t])
                    cj, cb = np.concatenate(cal['j']), np.concatenate(cal['b'])
                    mj, mb = np.concatenate(cal['mj']), np.concatenate(cal['mb'])
                    tj, tb = np.concatenate(tst['j']), np.concatenate(tst['b'])
                    for name, lam in (('scc', crc_lambda(cj, cb, alpha)), ('pt', crc_lambda(mj, mb, alpha)), ('ltt', ltt_lambda(cj, cb, alpha))):
                        com = tj <= lam
                        m['fcr_' + name].append((com & tb).mean()); m['cr_' + name].append(com.mean())
                        m['cond_' + name].append((com & tb).sum() / max(1, com.sum()))
                        if name != 'ltt': m['lam_' + name].append(lam)
                    m['p_bad'].append(tb.mean())
            nt = 500 * len(KS)
            r = dict(b=b, alpha=alpha, N=N, n_test=nt, p_bad_always_commit=float(np.mean(m['p_bad'])))
            for name in ('scc', 'pt', 'ltt'):
                f = float(np.mean(m['fcr_' + name])); lo, hi = wilson(f, nt)
                r.update({f'fcr_{name}': f, f'fcr_{name}_ci': [lo, hi], f'commit_{name}': float(np.mean(m['cr_' + name])),
                          f'cond_fcr_{name}': float(np.mean(m['cond_' + name]))})
                if name != 'ltt': r[f'lam_{name}'] = float(np.mean([x for x in m['lam_' + name] if np.isfinite(x)] or [-np.inf]))
            out['rows'].append(r)
json.dump(out, open(f'results/s1/commit_{args.scheme}_{POP}.json', 'w'), indent=1)
L = [f"commit/defer, scheme={args.scheme}, pop={POP}, pooled over checkpoints {KS}, n_test={500 * len(KS)}",
     '| bad if true pf > b | alpha | N | P(bad) always commit | SCC FCR [95%] | SCC commit rate | SCC P(bad given commit) | pointwise FCR [95%] | pointwise commit rate | pointwise P(bad given commit) | LTT-conditional: commit rate / P(bad given commit) |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
for r in out['rows']:
    L.append(f"| {r['b']:.2f} | {r['alpha']:.2f} | {r['N']} | {r['p_bad_always_commit']:.3f} | {r['fcr_scc']:.3f} [{r['fcr_scc_ci'][0]:.3f},{r['fcr_scc_ci'][1]:.3f}] | "
             f"{r['commit_scc']:.3f} | {r['cond_fcr_scc']:.3f} | {r['fcr_pt']:.3f} [{r['fcr_pt_ci'][0]:.3f},{r['fcr_pt_ci'][1]:.3f}] | {r['commit_pt']:.3f} | {r['cond_fcr_pt']:.3f} | {r['commit_ltt']:.3f} / {r['cond_fcr_ltt']:.3f} |")
open(f'results/s1/commit_{args.scheme}_{POP}.txt', 'w').write('\n'.join(L))
print('\n'.join(L))
