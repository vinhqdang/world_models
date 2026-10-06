"""S2: calibration analysis and fit of the SCC gate from the calibration collections (results/s2/calib_N{N}_m{m}.npz, seeds 1000+).

python experiments/s2_calfit.py --N 32 [--b 0.10]        -> results/s2/calfit_N32.npz, results/s2/calib_N32.txt

(1) Held-out coverage (episode-level 50/50 splits, 40 repeats) of the selected-plan band and the pointwise band (random final-population
    members) for the executed plan, for the fall-probability component and for the total planner cost.
(2) Offline commit / false-commit table of the two-stage rule as a function of the fall-probability budget b, on held-out episodes.
(3) Fit on ALL calibration states: sorted residual arrays used by the closed-loop gate (R1, R2 for the nominal alpha, pointwise analogues).
Stage-2 residuals are the residuals of the argmin-total-score candidate among the K=4 candidates, restricted to calibration states that
stage 1 does not commit at the nominal alpha (conformal validity conditional on escalation)."""
import argparse, sys, math
import numpy as np
sys.path.insert(0, '.')
from selwm.s2_scc import conf_quantile

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=32); ap.add_argument('--b', type=float, default=None)
ap.add_argument('--models', default='0,1,2'); ap.add_argument('--reps', type=int, default=40)
args = ap.parse_args()
MS = [int(x) for x in args.models.split(',')]
D = {m: dict(np.load(f'results/s2/calib_N{args.N}_m{m}.npz')) for m in MS}
ALPHAS = (0.05, 0.10)
L = []
P = lambda s='': (print(s), L.append(s))

def ep_id(d): return d['slot'] * 100000 + d['jep']

def derived(d):
    best = d['jc'].argmin(1); ar = np.arange(len(best))
    d['best'] = best
    d['R1'] = d['pT'] - d['pf']; d['Rp1'] = (d['pTm'] - d['pfm'])          # (n,), (n,4)
    d['R2'] = d['pTc'][ar, best] - d['pfc'][ar, best]; d['Rp2'] = d['pTc'] - d['pfc']      # (n,), (n,4)
    d['pf2'] = d['pfc'][ar, best]; d['pT2'] = d['pTc'][ar, best]
    d['RL1'] = d['L'] - d['j']; d['RLp1'] = d['Lm'] - d['jm']
    return d
for m in MS: derived(D[m])

def stage_rule(d_cal, d_te, alpha, b, band):
    """returns dict of rates on test data given calibration data"""
    if band == 'sel':
        q1 = conf_quantile(d_cal['R1'], alpha)
        U1c = d_cal['pf'] + q1
        esc = U1c > b
        q2 = conf_quantile(d_cal['R2'][esc], alpha) if esc.sum() > 30 else float('inf')
        U1 = d_te['pf'] + q1
        U2 = d_te['pf2'] + q2
    else:
        q1 = conf_quantile(d_cal['Rp1'].ravel(), alpha)
        U1c = d_cal['pf'] + q1
        esc = U1c > b
        q2 = conf_quantile(d_cal['Rp2'][esc].ravel(), alpha) if esc.sum() > 30 else float('inf')
        U1 = d_te['pf'] + q1
        U2 = d_te['pf2'] + q2
    c1 = U1 <= b; c2 = (~c1) & (U2 <= b)
    bad1 = d_te['pT'] > b; bad2 = d_te['pT2'] > b
    fc1 = c1 & bad1; fc2 = c2 & bad2
    n = len(U1)
    return dict(q1=q1, q2=q2, c1=c1.mean(), c2=c2.mean(), fc=(fc1.sum() + fc2.sum()) / n, fc1=fc1.mean(), fc2=fc2.mean(),
                cov1=np.mean(d_te['pT'] <= U1), cov1_esc=np.mean(d_te['pT2'][~c1] <= U2[~c1]) if (~c1).any() else np.nan)

def split(d, rng, frac=0.5):
    ids = np.unique(ep_id(d)); rng.shuffle(ids)
    cal_ids = set(ids[:int(len(ids) * frac)].tolist())
    msk = np.array([e in cal_ids for e in ep_id(d)])
    sub = lambda mk: {k: (v[mk] if (isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(mk)) else v) for k, v in d.items()}
    return sub(msk), sub(~msk)

def pool(ds):
    out = {}
    for k in ds[0]:
        v0 = ds[0][k]
        if isinstance(v0, np.ndarray) and v0.ndim >= 1 and len(v0) == len(ds[0]['pT']):
            out[k] = np.concatenate([d[k] for d in ds])
        else: out[k] = v0
    return out

N = args.N
P(f'S2 calibration summary, base planner N={N} (open-loop CEM, M=8, chained CRN, 3 iterations), wind 0.09, seeds 1000+m')
tot = 0
for m in MS:
    d = D[m]; n = len(d['pT']); tot += n
    P(f"model {m}: states {n}, episodes {len(np.unique(ep_id(d)))}, mean model fall prob {d['pf'].mean():.4f}, mean true fall prob {d['pT'].mean():.4f}, "
      f"P(pT>0.1) {np.mean(d['pT']>0.1):.3f}, P(pT>0.2) {np.mean(d['pT']>0.2):.3f}, mean j {d['j'].mean():.3f}, mean L {d['L'].mean():.3f}, realised F rate (resolved) {d['F'][d['F']>=0].mean():.4f}")
P(f'total calibration states {tot}')
P()
# ---------------------------------------------------------------- (1) coverage
P('(1) Held-out coverage of the executed plan (episode-level 50/50 split, %d repeats, per-model calibration then pooled over models)' % args.reps)
P(f"{'functional':28s} {'alpha':>5s} {'selected-plan band':>26s} {'pointwise band':>26s}  width sel / pw")
rng = np.random.default_rng(1)
for name, k1, k2 in (('fall probability', 'R1', 'Rp1'), ('total planner cost', 'RL1', 'RLp1')):
    for alpha in ALPHAS:
        cs, cp, ws, wp = [], [], [], []
        for rep in range(args.reps):
            hs, hp = 0, 0; n_te = 0; qs_, qp_ = [], []
            for m in MS:
                cal, te = split(D[m], rng)
                qs = conf_quantile(cal[k1], alpha); qp = conf_quantile(cal[k2].ravel(), alpha)
                hs += np.sum(te[k1] <= qs); hp += np.sum(te[k1] <= qp); n_te += len(te[k1]); qs_.append(qs); qp_.append(qp)
            cs.append(hs / n_te); cp.append(hp / n_te); ws.append(np.mean(qs_)); wp.append(np.mean(qp_))
        f = lambda a: f'{np.mean(a):.3f} [{np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f}]'
        P(f'{name:28s} {alpha:5.2f} {f(cs):>26s} {f(cp):>26s}  {np.mean(ws):.3f} / {np.mean(wp):.3f}   (nominal {1-alpha:.2f})')
P()
# ---------------------------------------------------------------- (2) offline commit table
P('(2) Two-stage rule, offline on held-out episodes: commit rate (stage 1, stage 2), false-commit rate FCR = P(commit and true H-step fall prob > b)')
P(f"{'b':>5s} {'alpha':>5s} {'band':>4s} {'q1':>7s} {'q2':>7s} {'commit s1':>9s} {'commit s2':>9s} {'defer':>6s} {'FCR':>6s}   (FCR target alpha)")
rng = np.random.default_rng(2)
for b in (0.05, 0.10, 0.15, 0.20, 0.30):
    for alpha in ALPHAS:
        for band in ('sel', 'pw'):
            acc = []
            for rep in range(args.reps // 2):
                rr = []
                for m in MS:
                    cal, te = split(D[m], rng); rr.append(stage_rule(cal, te, alpha, b, band))
                acc.append({k: np.mean([r[k] for r in rr]) for k in rr[0]})
            g = lambda k: np.mean([a[k] for a in acc])
            P(f"{b:5.2f} {alpha:5.2f} {band:>4s} {g('q1'):7.3f} {g('q2'):7.3f} {g('c1'):9.3f} {g('c2'):9.3f} {1-g('c1')-g('c2'):6.3f} {g('fc'):6.3f}")
P()
# ---------------------------------------------------------------- (3) fit
b = args.b
out = {}
if b is not None:
    out['b'] = np.array(b)
    P(f'(3) Fit on all calibration states with b = {b}')
    for m in MS:
        d = D[m]
        out[f'R1_m{m}'] = np.sort(d['R1']); out[f'pR1_m{m}'] = np.sort(d['Rp1'].ravel())
        for alpha in ALPHAS:
            tag = 'a%02d' % int(round(alpha * 100))
            q1 = conf_quantile(d['R1'], alpha); esc = (d['pf'] + q1) > b
            out[f'R2_{tag}_m{m}'] = np.sort(d['R2'][esc])
            qp1 = conf_quantile(d['Rp1'].ravel(), alpha); escp = (d['pf'] + qp1) > b
            out[f'pR2_{tag}_m{m}'] = np.sort(d['Rp2'][escp].ravel())
            P(f"model {m} alpha {alpha:.2f}: q1_sel {q1:.4f} (n {len(d['R1'])}) escalated {esc.mean():.3f} q2_sel {conf_quantile(d['R2'][esc], alpha):.4f} (n {esc.sum()}) | "
              f"q1_pw {qp1:.4f} escalated {escp.mean():.3f} q2_pw {conf_quantile(d['Rp2'][escp].ravel(), alpha):.4f}")
    np.savez(f'results/s2/calfit_N{N}.npz', **out)
open(f'results/s2/calib_N{N}.txt' if b is None else f'results/s2/calib_N{N}_fit.txt', 'w').write('\n'.join(L) + '\n')
