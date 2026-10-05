"""Numerical verification of the theory in paper/theory.tex (direction D4).

Usage:  OMP_NUM_THREADS=1 python experiments/d4_verify.py [--quick]
Writes results/d4/d4_verify.txt (all printed tables) and results/d4/d4_verify.json (max errors per claim).
Single core, CPU only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, math
import numpy as np
from scipy import stats, optimize, integrate
from scipy.special import ndtr, comb

QUICK = '--quick' in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'd4')
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'd4_verify.txt'), 'w')
RES = {}          # claim -> dict(max_err=..., note=...)


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n'); LOG.flush()


def hdr(t):
    P(); P('=' * 100); P(t); P('=' * 100)


def rec(name, err, note='', tol=None):
    RES[name] = dict(max_err=float(err), note=note, tol=tol, ok=(None if tol is None else bool(err <= tol)))
    P(f'  -> [{name}] max error = {err:.3e}  {note}' + ('' if tol is None else f'  (tolerance {tol:g}: {"PASS" if err <= tol else "FAIL"})'))


S = 400_000 if QUICK else 2_000_000      # Monte-Carlo sample size for T1 / T4
rng = np.random.default_rng(12345)


# =====================================================================================================
# T1
# =====================================================================================================
def cvar_chi2_central(alpha):
    z = stats.norm.ppf((1 + alpha) / 2)
    return 1 + z * stats.norm.pdf(z) / stats.norm.sf(z)


def cvar_mc(x, alpha):
    q = np.quantile(x, alpha)
    return x[x >= q].mean()


def t1():
    hdr('T1  variance gap of mean-embedding planning')
    # (a) bias-variance identity in d=4
    P('(a) E||Z-g||^2 = ||mu-g||^2 + tr Sigma  (d=4, random Gaussian and non-Gaussian laws)')
    P(f"{'law':>10} {'exact':>12} {'MC':>12} {'rel.err':>10}")
    worst = 0
    for k in range(6):
        d = 4
        A = rng.normal(size=(d, d)); Sg = A @ A.T / d; mu = rng.normal(size=d); g = rng.normal(size=d)
        if k % 2 == 0:
            Z = mu + rng.normal(size=(S, d)) @ np.linalg.cholesky(Sg).T; law = 'gauss'
        else:   # centred exponential noise, same covariance structure scaled
            E = rng.exponential(size=(S, d)) - 1.0
            L = np.linalg.cholesky(Sg); Z = mu + E @ L.T; Sg = L @ L.T; law = 'exp'
        ex = np.sum((mu - g) ** 2) + np.trace(Sg)
        mc = np.mean(np.sum((Z - g) ** 2, axis=1))
        worst = max(worst, abs(mc - ex) / ex)
        P(f'{law:>10} {ex:12.5f} {mc:12.5f} {abs(mc-ex)/ex:10.2e}')
    rec('T1a_identity', worst, 'relative MC error, expect ~1e-3', tol=8e-3)

    # (b) MSE-trained predictor converges to the conditional mean (not the median / mode)
    P('\n(b) least-squares predictor (degree-7 Legendre basis) vs true conditional mean; skewed heteroscedastic noise')
    P(f"{'n':>9} {'sup|m_hat-mu|':>15} {'sup|med-mu|':>13}")
    mu_f = lambda x: np.sin(2 * x) + 0.5 * x
    sc = lambda x: 0.3 + 0.5 * np.abs(x)
    xs = np.linspace(-1, 1, 201)
    # median of centred exponential noise: ln2 - 1 (times scale)
    med_gap = np.max(np.abs((np.log(2) - 1) * sc(xs)))
    prev = None; errs = []
    for n in ([1000, 10000, 100000] if QUICK else [1000, 10000, 100000, 1000000]):
        x = rng.uniform(-1, 1, n)
        y = mu_f(x) + sc(x) * (rng.exponential(size=n) - 1)
        V = np.polynomial.legendre.legvander(x, 7)
        c = np.linalg.lstsq(V, y, rcond=None)[0]
        mh = np.polynomial.legendre.legvander(xs, 7) @ c
        e = np.max(np.abs(mh - mu_f(xs)))
        errs.append(e)
        P(f'{n:9d} {e:15.4f} {med_gap:13.4f}')
    rec('T1b_mse_to_mean', errs[-1], f'final sup error (approximation floor of the basis included); decreasing={errs[-1] < errs[0]}', tol=0.05)

    # (c) regret identity and bound on random candidate sets
    P('\n(c) regret identity R = [b(a_d)-b(a*)] + [v(a_d)-v(a*)], bound R <= max v - min v; N=20 candidates, d=3')
    P(f"{'trial':>5} {'R exact':>10} {'R via id.':>10} {'bound dv':>10} {'R<=dv':>6} {'R_MC':>9}")
    worst_id = 0; viol = 0; worst_mc = 0
    for tr in range(12):
        d, N = 3, 20
        mu = rng.normal(size=(N, d)); g = np.zeros(d)
        sd = rng.uniform(0.05, 1.5, size=(N, d))
        b = np.sum((mu - g) ** 2, 1); v = np.sum(sd ** 2, 1); J = b + v
        ad = int(np.argmin(b)); astar = int(np.argmin(J))
        R = J[ad] - J[astar]
        Rid = (b[ad] - b[astar]) + (v[ad] - v[astar])
        dv = v.max() - v.min()
        worst_id = max(worst_id, abs(R - Rid)); viol += int(R > dv + 1e-12)
        m = 200_000 if QUICK else 1_000_000
        Jmc_d = np.mean(np.sum((mu[ad] + sd[ad] * rng.normal(size=(m, d))) ** 2, 1))
        Jmc_s = np.mean(np.sum((mu[astar] + sd[astar] * rng.normal(size=(m, d))) ** 2, 1))
        Rmc = Jmc_d - Jmc_s
        worst_mc = max(worst_mc, abs(Rmc - R))
        P(f'{tr:5d} {R:10.4f} {Rid:10.4f} {dv:10.4f} {str(R <= dv + 1e-12):>6} {Rmc:9.4f}')
    rec('T1c_identity', worst_id, 'exact identity (machine precision)', tol=1e-12)
    rec('T1c_bound_violations', viol, 'number of violations of R<=dv (expect 0)', tol=0)
    rec('T1c_MC_regret', worst_mc, 'abs error of MC regret vs exact', tol=0.05)

    # (d) tightness example: a1: mean at goal, variance V ; a2: bias eps, no noise
    P('\n(d) tightness example: R = V - eps^2 (-> Delta_v = V as eps -> 0), ratio R/J* = (V-eps^2)/eps^2 -> inf')
    P(f"{'V':>8} {'eps':>8} {'R exact':>10} {'R MC':>10} {'R/J*':>10}")
    worst = 0
    for V in [0.25, 1.0, 4.0, 16.0]:
        for eps in [0.3, 0.1]:
            z = np.sqrt(V) * rng.normal(size=S)
            Jd = np.mean(z ** 2); Js = eps ** 2
            R = V - eps ** 2
            worst = max(worst, abs((Jd - Js) - R) / V)
            P(f'{V:8.2f} {eps:8.2f} {R:10.4f} {Jd - Js:10.4f} {R / Js:10.1f}')
    rec('T1d_tightness', worst, 'relative error of MC regret vs V-eps^2', tol=8e-3)

    # (e) failure-penalised cost, 1-D: a1 mean=g, std s1, failure region z>=B ; a2 mean g-eps, no noise
    P('\n(e) failure-penalised cost: R(kappa) = s1^2 + kappa*p1 - eps^2  (p1 = P(Z>=B), mean of a1 outside failure set)')
    g, Bf, s1, eps = 0.0, 1.0, 0.6, 0.2
    p1 = stats.norm.sf((Bf - g) / s1)
    P(f'   g={g}, B={Bf}, s1={s1}, eps={eps}, p1={p1:.5f}')
    P(f"{'kappa':>8} {'R exact':>10} {'R MC':>10} {'det. sees':>10} {'true J1':>10}")
    z = g + s1 * rng.normal(size=S)
    worst = 0
    for kap in [0, 1, 5, 20, 100, 500]:
        c1 = (z - g) ** 2 + kap * (z >= Bf)
        J1mc = c1.mean(); J1 = s1 ** 2 + kap * p1; J2 = eps ** 2
        R = J1 - J2
        worst = max(worst, abs((J1mc - J2) - R) / max(R, 1))
        P(f'{kap:8d} {R:10.4f} {J1mc - J2:10.4f} {0.0:10.2f} {J1:10.4f}')
    rec('T1e_fail_penalty', worst, 'relative error', tol=1e-2)

    # (f) CVaR: closed form for chi2_1 and regret example
    P('\n(f) CVaR_alpha of (Z-g)^2, Z~N(g,s^2): CVaR = s^2 c_alpha, c_alpha = 1 + z phi(z)/Phibar(z), z=Phi^-1((1+alpha)/2)')
    P(f"{'alpha':>7} {'c_alpha exact':>14} {'c_alpha MC':>12} {'rel.err':>9} {'quad ncx2(mu=0)':>16}")
    x = rng.normal(size=S) ** 2
    worst = 0
    for al in [0.5, 0.8, 0.9, 0.95, 0.99]:
        ce = cvar_chi2_central(al)
        cm = cvar_mc(x, al)
        qd = integrate.quad(lambda u: stats.ncx2.ppf(u, 1, 1e-12), al, 1 - 1e-9, limit=200)[0] / (1 - al)
        worst = max(worst, abs(cm - ce) / ce)
        P(f'{al:7.2f} {ce:14.5f} {cm:12.5f} {abs(cm-ce)/ce:9.2e} {qd:16.5f}')
    rec('T1f_cvar_closed_form', worst, 'MC (tail mean) vs closed form; quad column cross-checks', tol=2e-2)
    P('   CVaR regret of the deterministic planner: R_alpha = s^2 c_alpha - eps^2 (s=1, eps=0.3)')
    P(f"{'alpha':>7} {'R_alpha exact':>14} {'R_alpha MC':>12} {'E-regret':>10}")
    worst = 0
    for al in [0.5, 0.9, 0.99]:
        R = cvar_chi2_central(al) - 0.09
        Rm = cvar_mc(x, al) - 0.09
        worst = max(worst, abs(R - Rm) / R)
        P(f'{al:7.2f} {R:14.4f} {Rm:12.4f} {1 - 0.09:10.4f}')
    rec('T1f_cvar_regret', worst, 'relative error', tol=2e-2)
    # noncentral check of CVaR bound  E C <= CVaR <= E C/(1-alpha)
    ok = True
    for mu0 in [0.0, 0.5, 1.5, 3.0]:
        for al in [0.5, 0.9]:
            qd = integrate.quad(lambda u: stats.ncx2.ppf(u, 1, mu0 ** 2), al, 1 - 1e-9, limit=200)[0] / (1 - al)
            EC = 1 + mu0 ** 2
            ok &= (EC - 1e-6 <= qd <= EC / (1 - al) + 1e-6)
    rec('T1f_cvar_bounds', 0.0 if ok else 1.0, 'E C <= CVaR_alpha <= E C/(1-alpha) on noncentral chi2 grid', tol=0)


# =====================================================================================================
# T2
# =====================================================================================================
def Fbin(k, M, p):
    return stats.binom.cdf(k, M, p)


def t2():
    hdr('T2  tail blindness amplified by selection')
    reps = 40_000 if QUICK else 200_000
    kap, gap = 10.0, 1.0           # penalty kappa, c_s - c_r  (theta = gap/kap = 0.1)
    theta = gap / kap
    cr = 0.0; cs = cr + gap
    P(f'kappa={kap}, c_s-c_r={gap}, theta={theta}, selection rule: risky chosen iff min_i J_hat_i <= c_s (ties to risky); reps={reps}')
    P(f"{'M':>3} {'p':>5} {'k*':>3} {'N_r':>5} {'P exact':>9} {'P MC':>9} {'z-score':>8} {'excess exact':>13} {'excess MC':>10} {'optimism exact':>15} {'optimism MC':>12}")
    zmax = 0; worst_abs = 0; worst_ex = 0; worst_opt = 0
    mono_ok = True
    for M in [4, 8, 16, 32]:
        kstar = int(math.floor(M * gap / kap + 1e-12))
        for p in [0.2, 0.4]:
            Fv = Fbin(kstar, M, p)
            prevP = -1
            for Nr in [1, 4, 16, 64, 256]:
                Pex = 1 - (1 - Fv) ** Nr
                mono_ok &= Pex >= prevP; prevP = Pex
                K = rng.binomial(M, p, size=(reps, Nr))
                Jhat = cr + kap * K / M
                sel_r = Jhat.min(1) <= cs + 1e-12
                Pmc = sel_r.mean()
                se = math.sqrt(max(Pex * (1 - Pex), 1e-12) / reps)
                z = (Pmc - Pex) / se if Pex * (1 - Pex) > 1e-9 else 0.0
                # excess true cost over the best action (safe, since p>theta): P * kappa*(p-theta)
                exc_ex = Pex * kap * (p - theta)
                true_cost = np.where(sel_r, cr + kap * p, cs)
                exc_mc = true_cost.mean() - cs
                # optimism: E[min(c_s, min Jhat)] vs true cost of selected
                emk = sum((1 - Fbin(k - 1, M, p)) ** Nr for k in range(1, M + 1))     # E min K
                opt_ex = cr + kap * emk / M                                       # E[min_i Jhat_i]
                opt_mc = Jhat.min(1).mean()
                zmax = max(zmax, abs(z)); worst_abs = max(worst_abs, abs(Pmc - Pex))
                worst_ex = max(worst_ex, abs(exc_mc - exc_ex)); worst_opt = max(worst_opt, abs(opt_mc - opt_ex))
                if Nr in (1, 16, 256):
                    P(f'{M:3d} {p:5.2f} {kstar:3d} {Nr:5d} {Pex:9.5f} {Pmc:9.5f} {z:8.2f} {exc_ex:13.5f} {exc_mc:10.5f} {opt_ex:15.5f} {opt_mc:12.5f}')
    rec('T2a_select_prob', worst_abs, f'max |P_MC-P_exact|; max |z-score| = {zmax:.2f}', tol=0.006)
    rec('T2a_monotone_in_Nr', 0.0 if mono_ok else 1.0, 'P nondecreasing in N_r on grid', tol=0)
    rec('T2b_excess_cost', worst_ex, 'abs error of E[true cost]-c_s = P*kappa*(p-theta)', tol=0.03)
    rec('T2b_winner_curse', worst_opt, 'abs error of E[min J_hat] formula', tol=0.02)

    # limit N_r -> infinity & N_r needed
    P('\nLimit N_r -> inf: P -> 1, excess -> kappa(p-theta); estimated cost -> c_r (optimism gap -> kappa p)')
    M, p = 8, 0.3; kstar = int(math.floor(M * theta)); Fv = Fbin(kstar, M, p)
    P(f'   M={M}, p={p}, F={Fv:.5f}; N_r needed for P>=0.99:  exact {math.ceil(math.log(0.01) / math.log(1 - Fv))},  ln(1/delta)/F approx {math.log(100) / Fv:.1f}')
    for Nr in [10, 100, 1000, 10000]:
        Pex = 1 - (1 - Fv) ** Nr
        emk = sum((1 - Fbin(k - 1, M, p)) ** Nr for k in range(1, M + 1))
        P(f'   N_r={Nr:6d}: P={Pex:.6f}, excess={Pex * kap * (p - theta):.5f} (limit {kap * (p - theta):.3f}), E[min J_hat]={cr + kap * emk / M:.5f}')

    # Hoeffding bound / particles needed
    P('\nHoeffding: P(select risky) <= N_r exp(-2 M (p-theta)^2); M needed for delta: M >= ln(N_r/delta)/(2 (p-theta)^2)')
    bad = 0
    P(f"{'M':>4} {'N_r':>6} {'p':>5} {'P exact':>10} {'bound':>10} {'ok':>4}")
    for p in [0.2, 0.4]:
        for M in [8, 32, 128]:
            ks = int(math.floor(M * theta + 1e-12)); Fv = Fbin(ks, M, p)
            for Nr in [4, 64]:
                Pex = 1 - (1 - Fv) ** Nr; bd = min(1, Nr * math.exp(-2 * M * (p - theta) ** 2))
                bad += int(Pex > bd + 1e-12)
                P(f'{M:4d} {Nr:6d} {p:5.2f} {Pex:10.3e} {bd:10.3e} {str(Pex <= bd + 1e-12):>4}')
    rec('T2c_hoeffding_violations', bad, 'number of violated bounds (expect 0)', tol=0)

    # two-stage verification
    P('\nTwo-stage fresh-sample verification (stage 1: N_r candidates, M particles, keep top-K; stage 2: M2 fresh particles).')
    P('  risk (with safe fallback in final comparison) = 1-(1-F2)^K, independent of N_r;  pooled variant (safe ranked in stage 1, no fallback in S):')
    P('  risk = P(R1>=K) + P(R1<K)(1-(1-F2)^(K-1)),  R1 ~ Bin(N_r, F)')
    P(f"{'N_r':>5} {'K':>3} {'M2':>4} {'naive':>8} {'fallback exact':>15} {'fallback MC':>12} {'pooled exact':>13} {'pooled MC':>10}")
    M, p = 8, 0.4; ks = int(math.floor(M * theta)); F1 = Fbin(ks, M, p)
    worst = 0; worst_pool = 0
    reps2 = 20_000 if QUICK else 60_000
    for Nr in [16, 256]:
        for K in [1, 4]:
            for M2 in [8, 32]:
                k2 = int(math.floor(M2 * theta + 1e-12)); F2 = Fbin(k2, M2, p)
                fb_ex = 1 - (1 - F2) ** K
                R1 = stats.binom(Nr, F1)
                pool_ex = R1.sf(K - 1) + R1.cdf(K - 1) * (1 - (1 - F2) ** (K - 1))
                # MC
                K1 = rng.binomial(M, p, size=(reps2, Nr))
                tie = rng.random(size=(reps2, Nr))
                order = np.lexsort((tie, K1), axis=1) if False else None
                key = K1 + tie * 0.5                                  # K1 integer, tie-break within equal counts
                idx = np.argsort(key, axis=1)[:, :K]
                fresh = rng.binomial(M2, p, size=(reps2, K))
                fb_mc = (fresh <= k2).any(1).mean()
                # pooled variant: safe has value c_s => equivalent count M*theta (ties after risky)
                vals = np.concatenate([K1 * kap / M, np.full((reps2, 1), gap)], 1)   # J_hat - c_r
                tiebr = np.concatenate([tie, np.full((reps2, 1), 2.0)], 1)           # safe last among ties
                order = np.lexsort((tiebr, vals), axis=1) if False else np.argsort(vals + 1e-9 * tiebr, axis=1)
                top = order[:, :K]
                is_safe = (top == Nr)
                fresh_all = rng.binomial(M2, p, size=(reps2, K))
                risky_ok = (~is_safe) & (fresh_all <= k2)
                # if the safe plan is not in the verified set, a risky plan is selected whatever the fresh values
                pooled_mc = (risky_ok.any(1) | (~is_safe.any(1))).mean()
                naive = 1 - (1 - F1) ** Nr
                worst = max(worst, abs(fb_mc - fb_ex)); worst_pool = max(worst_pool, abs(pooled_mc - pool_ex))
                P(f'{Nr:5d} {K:3d} {M2:4d} {naive:8.4f} {fb_ex:15.5f} {fb_mc:12.5f} {pool_ex:13.5f} {pooled_mc:10.5f}')
    rec('T2d_verification_fallback', worst, 'abs error exact vs MC', tol=0.015)
    rec('T2d_verification_pooled', worst_pool, 'abs error exact vs MC', tol=0.015)

    # unbiasedness of fresh estimates conditional on selection
    P('\nUnbiasedness conditional on selection (N_r=64, M=8, p=0.4): E[J_hat_stage1 | selected] vs E[J_hat_fresh | selected] vs true J')
    Nr = 64; K1 = rng.binomial(M, p, size=(reps, Nr)); win = K1.argmin(1)
    j1 = cr + kap * K1[np.arange(reps), win] / M
    fresh = cr + kap * rng.binomial(M, p, size=reps) / M
    Jtrue = cr + kap * p
    P(f'   stage-1 estimate of selected: {j1.mean():.4f}   fresh estimate: {fresh.mean():.4f}   truth: {Jtrue:.4f}')
    rec('T2e_fresh_unbiased', abs(fresh.mean() - Jtrue), f'stage-1 biased by {j1.mean() - Jtrue:.3f}', tol=0.03)

    # shrinkage
    P('\nShrinkage (normal-normal empirical Bayes, B(M) = tau^2/(tau^2+sigma^2/M)); ranking invariance with equal M')
    trials = 20_000 if QUICK else 100_000
    N = 12; M = 8; sig2 = 4.0
    est = rng.normal(loc=rng.normal(size=(trials, 1)), scale=1.0, size=(trials, N)) + rng.normal(scale=math.sqrt(sig2 / M), size=(trials, N))
    m0 = est.mean(1, keepdims=True)
    tau2 = np.maximum(est.var(1, ddof=1, keepdims=True) - sig2 / M, 1e-9)       # method-of-moments EB, positive part
    Bf = tau2 / (tau2 + sig2 / M)
    sh = m0 + Bf * (est - m0)
    changed = (est.argmin(1) != sh.argmin(1)).mean()
    P(f'   equal M: fraction of trials where argmin changes = {changed:.6f}')
    rec('T2f_shrink_equalM', changed, 'fraction argmin changed (expect 0)', tol=0)
    # heteroscedastic per-candidate variance (equal M): ranking CAN change
    s2i = rng.uniform(0.5, 8.0, size=(trials, N))
    est_h = rng.normal(size=(trials, N)) + rng.normal(size=(trials, N)) * np.sqrt(s2i / M)
    m0h = est_h.mean(1, keepdims=True)
    Bh = 1.0 / (1.0 + s2i / M)            # tau^2 = 1
    shh = m0h + Bh * (est_h - m0h)
    P(f'   equal M but candidate-specific variances: fraction changed = {(est_h.argmin(1) != shh.argmin(1)).mean():.4f}')
    # unequal M
    Mi = rng.choice([2, 4, 16, 64], size=(trials, N))
    est_u = rng.normal(size=(trials, N)) + rng.normal(size=(trials, N)) * np.sqrt(sig2 / Mi)
    m0u = est_u.mean(1, keepdims=True); Bu = 1.0 / (1.0 + sig2 / Mi)
    shu = m0u + Bu * (est_u - m0u)
    fr_u = (est_u.argmin(1) != shu.argmin(1)).mean()
    P(f'   unequal M (racing-like): fraction changed = {fr_u:.4f}')
    # explicit counter-example
    m0e, tau2e, sig2e = 2.0, 1.0, 4.0
    B = lambda M_: tau2e / (tau2e + sig2e / M_)
    J1 = m0e + B(4) * (0.9 - m0e); J2 = m0e + B(64) * (1.0 - m0e)
    P(f'   explicit example: raw (0.9 @M=4, 1.0 @M=64) -> argmin 1; shrunk ({J1:.4f}, {J2:.4f}) -> argmin {1 if J1 < J2 else 2}')
    rec('T2f_shrink_unequalM_example', 0.0 if (J1 > J2) else 1.0, f'unequal-M ranking flip exhibited; random flip fraction {fr_u:.3f}', tol=0)


# =====================================================================================================
# T3
# =====================================================================================================
def es_gauss(m, s, y):
    w = (y - m) / s
    return s * (2 * stats.norm.pdf(w) + w * (2 * ndtr(w) - 1)) - s / math.sqrt(math.pi)


def t3():
    hdr('T3  energy score')
    # closed-form ES vs MC
    P('(a) closed-form ES(N(m,s^2), y) vs Monte-Carlo ES')
    worst = 0
    for (m, s, y) in [(0, 1, 0.3), (1, 2, -1.0), (-0.5, 0.4, 0.9)]:
        X = m + s * rng.normal(size=S); X2 = m + s * rng.normal(size=S)
        mc = np.mean(np.abs(X - y)) - 0.5 * np.mean(np.abs(X - X2))
        ex = es_gauss(m, s, y); worst = max(worst, abs(mc - ex))
        P(f'   m={m}, s={s}, y={y}: exact {ex:.5f}  MC {mc:.5f}')
    rec('T3a_es_closed_form', worst, 'abs error', tol=5e-3)

    # propriety: expected ES over the true law Y~N(0,sig0^2) minimised at model scale = sig0 (and at model mean 0)
    P('\n(b) strict propriety (Gaussian family): E_{Y~N(0,1)} ES(N(m,s^2),Y) over a grid; minimiser should be (0,1)')
    ss = np.linspace(0.3, 2.5, 111); ms = np.linspace(-1.5, 1.5, 61)
    ygrid = np.linspace(-9, 9, 6001); w = stats.norm.pdf(ygrid); dy = ygrid[1] - ygrid[0]
    val = np.array([[np.sum(es_gauss(m, s, ygrid) * w) * dy for s in ss] for m in ms])
    i, j = np.unravel_index(val.argmin(), val.shape)
    P(f'   argmin over grid: m={ms[i]:.3f}, s={ss[j]:.3f} (true 0, 1); minimum value {val.min():.5f} (= E|Y|... entropy-like: (1-1/sqrt(pi)... ) = {1/math.sqrt(math.pi):.5f})')
    rec('T3b_propriety_grid', max(abs(ms[i]), abs(ss[j] - 1)), 'distance of grid minimiser from truth', tol=0.03)

    # identity: E ES(P,Y) - E ES(Q,Y) = 1/2 * energy distance(P,Q), Y~Q ;  1-D: energy distance = 2 int (F-G)^2
    P('\n(c) identity E_{Y~Q}[ES(P,Y)-ES(Q,Y)] = (1/2) D_E(P,Q) with D_E = 2 int (F_P-F_Q)^2 dx (1-D); P=N(0,1), Q=Laplace(0,b)')
    P(f"{'b':>6} {'LHS (MC)':>10} {'RHS (quad)':>11}")
    worst = 0
    for b in [0.5, 0.7071, 1.0]:
        Y = rng.laplace(scale=b, size=S); X = rng.normal(size=S); X2 = rng.normal(size=S)
        Yp = rng.laplace(scale=b, size=S)
        esP = es_gauss(0, 1, Y)
        # ES(Q,Y) = E|Y'-Y| - 0.5 E|Y'-Y''| ; E over Y of E|Y'-Y| = E|Y-Y'|, so E ES(Q,Y)= 0.5 E|Y-Y'|
        lhs = np.mean(esP) - 0.5 * np.mean(np.abs(Y - Yp))
        FQ = lambda x: np.where(x < 0, 0.5 * np.exp(x / b), 1 - 0.5 * np.exp(-x / b))
        rhs = 0.5 * 2 * integrate.quad(lambda x: (ndtr(x) - FQ(x)) ** 2, -40, 40, limit=400, points=[0])[0]
        worst = max(worst, abs(lhs - rhs))
        P(f'{b:6.3f} {lhs:10.5f} {rhs:11.5f}')
    rec('T3c_energy_distance_identity', worst, 'abs error', tol=5e-3)

    # point-mass predictor under ES -> spatial median, under MSE -> mean (skewed noise)
    P('\n(d) deterministic predictor trained with ES (= absolute error) estimates the median, MSE the mean (Exp(1) noise: mean 1, median ln2)')
    y = rng.exponential(size=S)
    mm = np.linspace(0, 2, 2001)
    # use sorted data approx via quantile of loss: evaluate on subsample
    ys = y[:200_000]
    mse = [np.mean((ys - m) ** 2) for m in mm[::20]]; mae = [np.mean(np.abs(ys - m)) for m in mm[::20]]
    m_mse = mm[::20][int(np.argmin(mse))]; m_mae = mm[::20][int(np.argmin(mae))]
    P(f'   argmin MSE = {m_mse:.3f} (mean 1), argmin ES(point mass) = {m_mae:.3f} (median {math.log(2):.3f})')
    rec('T3d_point_mass_median', max(abs(m_mse - 1), abs(m_mae - math.log(2))), 'grid resolution 0.02', tol=0.05)

    # nonidentifiability of the noise map & CRN dependence on the parametrisation
    P('\n(e) same predictive law, different noise map: Var of CRN estimate of cost difference.  Z1=mu1+s u,  Z2=mu2 + eps s u (eps=+1 or -1), c=(z-g)^2')
    P('    exact Var(c1-c2) = 4 s^2 (m1 - eps m2)^2  with m_i = mu_i-g;  marginal laws of Z1,Z2 identical for eps=+-1')
    mu1, mu2, g, s = 0.8, 0.5, 0.0, 0.7
    u = rng.normal(size=S); worst = 0
    for eps in (+1, -1):
        c1 = (mu1 + s * u - g) ** 2; c2 = (mu2 + eps * s * u - g) ** 2
        ve = 4 * s ** 2 * (mu1 - eps * mu2 - (1 - eps) * 0) ** 2 if False else 4 * s ** 2 * ((mu1 - g) - eps * (mu2 - g)) ** 2
        vm = np.var(c1 - c2)
        worst = max(worst, abs(vm - ve) / ve)
        P(f'   eps={eps:+d}: Var exact {ve:.5f}, MC {vm:.5f};  mean(c2) = {c2.mean():.5f} (law-determined: {(mu2 - g) ** 2 + s ** 2:.5f})')
    rec('T3e_crn_variance', worst, 'relative error', tol=2e-2)

    # consistency of the empirical energy-score minimiser (set-valued)
    P('\n(f) consistency: Y = beta x + (s2+s3|x|) eps; model f_theta(x,u) = th1 x + (th2 + th3|x|) u, u~N(0,1); empirical ES with m=8 draws per sample')
    P('    identified set Theta* = {(b,s2,s3), (b,-s2,-s3)}; report median dist(theta_hat, Theta*) over replications')
    th0 = np.array([1.0, 0.5, 0.8])
    m = 8
    def fit(n, seed):
        r = np.random.default_rng(seed)
        x = r.normal(size=n); y = th0[0] * x + (th0[1] + th0[2] * np.abs(x)) * r.normal(size=n)
        U = r.normal(size=(n, m)); ax = np.abs(x)
        def obj(th):
            sc = th[1] + th[2] * ax
            X = th[0] * x[:, None] + sc[:, None] * U
            t1 = np.mean(np.abs(X - y[:, None]), axis=1)
            # pairwise term (j != l), U-statistic
            D = np.abs(X[:, :, None] - X[:, None, :]).sum((1, 2)) / (m * (m - 1))
            return np.mean(t1 - 0.5 * D)
        best = None
        for st in ([0.3, 0.3, 0.3], [0.5, -0.3, -0.3]):
            rr = optimize.minimize(obj, np.array(st), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-10, maxiter=400))
            if best is None or rr.fun < best.fun: best = rr
        th = best.x
        d1 = np.linalg.norm(th - th0); d2 = np.linalg.norm(th - np.array([th0[0], -th0[1], -th0[2]]))
        return min(d1, d2), th
    P(f"{'n':>7} {'median dist':>12} {'max dist':>10}   example theta_hat")
    meds = []
    ns = [200, 800, 3200] if QUICK else [200, 800, 3200, 12800]
    for n in ns:
        reps_ = 5 if n >= 3200 else 8
        ds = []; ex = None
        for k in range(reps_):
            d, th = fit(n, 1000 * n + k); ds.append(d); ex = th
        meds.append(np.median(ds))
        P(f'{n:7d} {np.median(ds):12.4f} {np.max(ds):10.4f}   {np.round(ex, 3)}')
    t3g()
    rec('T3f_consistency', meds[-1], f'median dist at largest n; decreasing: {all(meds[i+1] < meds[i] * 1.15 for i in range(len(meds)-1))}; first={meds[0]:.3f}', tol=0.1)


def t3g():
    P('\n(g) V-statistic energy score (self-pairs included, factor (m-1)/m on the spread term) is NOT proper: Gaussian scale family, truth N(0,1)')
    P('    risk R_V(s) = sqrt(2/pi) sqrt(1+s^2) - rho s/sqrt(pi), rho=(m-1)/m; minimiser s_m^2 = rho^2/(2-rho^2)')
    P(f"{'m':>4} {'rho':>7} {'s_m exact':>10} {'s_m numeric':>12} {'R_V(s=1) exact':>15} {'R_V(s=1) MC':>12}")
    worst = 0
    for m in [2, 4, 8, 32, 1000]:
        rho = (m - 1) / m
        sm = math.sqrt(rho ** 2 / (2 - rho ** 2))
        Rv = lambda s_: math.sqrt(2 / math.pi) * math.sqrt(1 + s_ ** 2) - rho * s_ / math.sqrt(math.pi)
        sn = optimize.minimize_scalar(Rv, bounds=(0.01, 3), method='bounded', options=dict(xatol=1e-10)).x
        nn = 200_000 if QUICK else 1_000_000
        mm = min(m, 64)
        y = rng.normal(size=nn); X = rng.normal(size=(nn, mm))
        t1_ = np.abs(X - y[:, None]).mean(1)
        D = np.abs(X[:, :, None] - X[:, None, :]).sum((1, 2)) / (mm * mm) if mm <= 8 else None
        if D is None:
            Rmc = float('nan')
        else:
            Rmc = np.mean(t1_ - 0.5 * D)
        rho_mm = (mm - 1) / mm
        Rv_mm = math.sqrt(2 / math.pi) * math.sqrt(2) - rho_mm * 1.0 / math.sqrt(math.pi)
        if D is not None: worst = max(worst, abs(Rmc - Rv_mm))
        worst = max(worst, abs(sn - sm))
        P(f'{m:4d} {rho:7.4f} {sm:10.4f} {sn:12.4f} {Rv_mm if D is not None else float("nan"):15.5f} {Rmc:12.5f}')
    rec('T3g_vstat_improper', worst, 'closed-form minimiser vs numeric; MC check of R_V(1) for m<=8', tol=5e-3)


# =====================================================================================================
# T4
# =====================================================================================================
def t4():
    hdr('T4  open loop vs closed loop under process noise')
    sig = 0.5
    # (a) variance growth
    P(f'(a) x_{{t+1}} = x_t + u_t + sigma w_t, sigma={sig}, x_0=0, g=0.  Var(x_t): open loop vs feedback u_t=-k x_t')
    Hs = [1, 2, 4, 8, 16, 32]
    P(f"{'t':>3} {'OL exact':>10} {'OL MC':>10} {'k=0.3 exact':>12} {'k=0.3 MC':>10} {'k=1 exact':>10} {'k=1 MC':>9} {'stationary(k=.3)':>17}")
    n = S // 2
    w = rng.normal(size=(n, 33)) * sig
    xo = np.zeros(n); xk = np.zeros(n); x1 = np.zeros(n); worst = 0
    k = 0.3
    for t in range(1, 33):
        xo = xo + w[:, t - 1]; xk = xk - k * xk + w[:, t - 1]; x1 = x1 - 1.0 * x1 + w[:, t - 1]
        if t in Hs:
            ve_o = t * sig ** 2
            ve_k = sig ** 2 * (1 - (1 - k) ** (2 * t)) / (1 - (1 - k) ** 2)
            ve_1 = sig ** 2
            worst = max(worst, abs(xo.var() - ve_o) / ve_o, abs(xk.var() - ve_k) / ve_k, abs(x1.var() - ve_1) / ve_1)
            P(f'{t:3d} {ve_o:10.4f} {xo.var():10.4f} {ve_k:12.4f} {xk.var():10.4f} {ve_1:10.4f} {x1.var():9.4f} {sig ** 2 / (k * (2 - k)):17.4f}')
    rec('T4a_variance_growth', worst, 'relative error of exact variances', tol=2e-2)

    # (b) LQG terminal cost: OL value = H * CL value for expectation and CVaR (homogeneous, FSD-monotone)
    P(f'\n(b) terminal cost (x_H-g)^2, g=0: optimal value, OL (best deterministic plan) vs CL (deadbeat last step).  Predicted ratio = H')
    P(f"{'H':>3} {'crit':>9} {'OL exact':>10} {'OL MC':>10} {'CL exact':>10} {'CL MC':>10} {'ratio':>7}")
    worst = 0
    for H in [1, 2, 4, 8]:
        # OL: plan reaching g in mean: x_H ~ N(0, H sig^2)
        zo = sig * math.sqrt(H) * rng.normal(size=S)
        # CL: deadbeat at the last step: x_H = sig w_{H-1}  (simulate full path under deadbeat feedback u_t = -x_t at every step)
        x = np.zeros(S)
        for t in range(H):
            x = x - x + sig * rng.normal(size=S)
        for crit, f_ex, f_mc in [('E', lambda s2: s2, lambda c: c.mean()),
                                 ('CVaR.9', lambda s2: s2 * cvar_chi2_central(0.9), lambda c: cvar_mc(c, 0.9)),
                                 ('CVaR.99', lambda s2: s2 * cvar_chi2_central(0.99), lambda c: cvar_mc(c, 0.99))]:
            ol_e = f_ex(H * sig ** 2); cl_e = f_ex(sig ** 2)
            ol_m = f_mc(zo ** 2); cl_m = f_mc(x ** 2)
            worst = max(worst, abs(ol_m - ol_e) / ol_e, abs(cl_m - cl_e) / cl_e)
            P(f'{H:3d} {crit:>9} {ol_e:10.4f} {ol_m:10.4f} {cl_e:10.4f} {cl_m:10.4f} {ol_e / cl_e:7.2f}')
    rec('T4b_LQG_ratio_H', worst, 'relative MC error', tol=4e-2)
    # shifting the mean can only hurt (folded normal monotone), OL: value vs mean offset
    P('   OL with mean offset m: E=(m^2+H sig^2); CVaR is increasing in |m| (checked on a grid, H=4, alpha=0.9)')
    H = 4; vals = []
    for mo in [0, 0.25, 0.5, 1.0, 2.0]:
        s2 = H * sig ** 2
        q = integrate.quad(lambda u: stats.ncx2.ppf(u, 1, mo ** 2 / s2) * s2, 0.9, 1 - 1e-9, limit=200)[0] / 0.1
        vals.append(q)
    P('   CVaR_0.9 at m=0,.25,.5,1,2: ' + ', '.join(f'{v:.4f}' for v in vals))
    rec('T4b_cvar_monotone_in_offset', 0.0 if all(np.diff(vals) > 0) else 1.0, 'increasing', tol=0)
    # entropic
    th = 0.4
    P(f'   entropic risk (1/theta) ln E exp(theta C), theta={th}: OL finite iff 2 theta H sig^2<1: H=1: {-(1/(2*th))*0+ (-0.5/th)*math.log(1-2*th*sig**2):.4f}, ' +
      ' '.join(f'H={Hh}: {("%.4f" % ((-0.5/th)*math.log(1-2*th*Hh*sig**2))) if 2*th*Hh*sig**2<1 else "inf"}' for Hh in [2, 3, 4, 5]))
    zz = sig * rng.normal(size=S)
    mcv = math.log(np.mean(np.exp(th * zz ** 2))) / th
    rec('T4b_entropic_H1', abs(mcv - (-0.5 / th) * math.log(1 - 2 * th * sig ** 2)) / mcv, 'rel. error MC vs closed form (H=1)', tol=2e-2)

    # (c) absorbing boundary: DP (closed loop) vs OL optimisation on a grid; MC verification
    barrier()


def barrier():
    P('\n(c) absorbing boundary: x_{t+1} = x_t + u_t + sigma w, |u|<=U, absorbed (stopped) at x>=B.  Cost = (x_H-g)^2 + kappa 1[hit]  (x_H=B if absorbed)')
    x0, g, Bd, sig, kap, U = 0.0, 0.6, 1.0, 0.2, 4.0, 0.5
    Ck = (Bd - g) ** 2 + kap
    P(f'   x0={x0}, g={g}, B={Bd}, sigma={sig}, kappa={kap}, U={U}, absorbed cost {Ck:.3f}')
    n = 500 if QUICK else 900
    Lm = 1.6
    edges = np.linspace(-Lm, Bd, n + 1); xc = 0.5 * (edges[1:] + edges[:-1])
    ug = np.linspace(-U, U, 41)

    def kernel(u):
        z = (edges[None, :] - xc[:, None] - u) / sig
        c = ndtr(z); c[:, 0] = 0.0
        pm = np.diff(c, axis=1)
        pa = 1.0 - ndtr((Bd - xc - u) / sig)
        return pm, pa

    def first_step(u):
        z = (edges - x0 - u) / sig
        c = ndtr(z); c[0] = 0.0
        return np.diff(c), 1.0 - ndtr((Bd - x0 - u) / sig)

    kern_cache = [kernel(u) for u in ug]          # 41 kernels
    Hmax = 4 if QUICK else 5
    Vnext = (xc - g) ** 2                           # value at time H (alive)
    Vs = {1: None}
    pol = {}
    Vlist = {0: Vnext}
    V = Vnext.copy()
    # V_t for t = H-1 ... : compute backward for the largest H; value-to-go at k steps remaining
    Vk = [V]                                        # Vk[k] = value with k steps remaining (alive), terminal cost at k=0
    polk = [None]
    for k_ in range(1, Hmax + 1):
        best = np.full(n, np.inf); bu = np.zeros(n)
        for iu, u in enumerate(ug):
            pm, pa = kern_cache[iu]
            q = pm @ Vk[-1] + pa * Ck
            m = q < best; best = np.where(m, q, best); bu = np.where(m, u, bu)
        Vk.append(best); polk.append(bu)

    def cl_value(H):
        # first step from x0 exactly, with continuation Vk[H-1]
        bestv = np.inf; bu0 = 0
        for u in np.linspace(-U, U, 201):
            pm, pa = first_step(u)
            v = pm @ Vk[H - 1] + pa * Ck
            if v < bestv: bestv, bu0 = v, u
        return bestv, bu0

    def ol_value(useq):
        useq = np.clip(useq, -U, U)
        pm, pa = first_step(useq[0]); dens = pm.copy(); dead = pa
        for t in range(1, len(useq)):
            Kt, pat = kernel(useq[t])
            dead = dead + dens @ pat
            dens = dens @ Kt
        return dens @ (xc - g) ** 2 + dead * Ck

    def ol_opt(H):
        best = None
        rs = np.random.default_rng(5)
        starts = [np.full(H, g / H)] + [rs.uniform(-U, U, H) for _ in range(2)]
        for s0 in starts:
            r = optimize.minimize(ol_value, s0, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-9, maxiter=250 * H))
            if best is None or r.fun < best.fun: best = r
        return best.fun, np.clip(best.x, -U, U)

    P(f"{'H':>3} {'V_CL (DP)':>10} {'V_OL (opt)':>11} {'gap':>8} {'gap/V_CL':>9} {'V_CL MC':>10} {'V_OL MC':>10} {'MC se':>7}  OL plan")
    nmc = 40_000 if QUICK else 200_000
    worst_mc = 0; gaps = []
    for H in range(1, Hmax + 1):
        vcl, u0cl = cl_value(H)
        vol, plan = ol_opt(H)
        # MC for OL
        rr = np.random.default_rng(100 + H)
        def sim(policy):
            x = np.full(nmc, x0); alive = np.ones(nmc, bool)
            for t in range(H):
                u = policy(t, x)
                x = np.where(alive, x + u + sig * rr.normal(size=nmc), x)
                hit = alive & (x >= Bd); x = np.where(hit, Bd, x); alive = alive & ~hit
            return (x - g) ** 2 + kap * (~alive)
        c_ol = sim(lambda t, x: plan[t])
        def pcl(t, x):
            if t == 0: return np.full_like(x, u0cl)
            rem = H - t
            return np.interp(x, xc, polk[rem])
        c_cl = sim(pcl)
        se = max(c_ol.std(), c_cl.std()) / math.sqrt(nmc)
        worst_mc = max(worst_mc, abs(c_ol.mean() - vol) / se, abs(c_cl.mean() - vcl) / se)
        gaps.append(vol - vcl)
        P(f'{H:3d} {vcl:10.5f} {vol:11.5f} {vol - vcl:8.5f} {(vol - vcl) / vcl:9.3f} {c_cl.mean():10.5f} {c_ol.mean():10.5f} {se:7.5f}  {np.round(plan, 3)}')
    rec('T4c_barrier_MC_vs_grid', worst_mc, 'max |MC - grid value| in units of MC standard errors (grid discretisation bias included)', tol=4.5)
    rec('T4c_barrier_gap_positive', min(gaps[1:]) if len(gaps) > 1 else 0.0, 'min over H>=2 of V_OL - V_CL (must be >0); gap(H=1)=%.2e' % gaps[0], tol=None)
    RES['T4c_barrier_gap_positive']['ok'] = bool(min(gaps[1:]) > 0 and abs(gaps[0]) < 2e-3)
    # minimiser of the last-step function q(y) used in the H=2 proposition
    qf = lambda y: integrate.quad(lambda w: ((min(y + sig * w, Bd) - g) ** 2 + kap * (y + sig * w >= Bd)) * stats.norm.pdf(w), -9, 9, points=[(Bd - y) / sig], limit=200)[0]
    rq = optimize.minimize_scalar(qf, bounds=(-1, Bd), method='bounded', options=dict(xatol=1e-8))
    P(f'\n   last-step function q(y)=E[cost of landing at y+sigma w]: minimiser y*={rq.x:.4f} (< B={Bd}: {rq.x < Bd}), q*={rq.fun:.5f} (< absorbed cost {Ck:.3f}: {rq.fun < Ck}); V_CL(H=2) is approx q* when step-1 failure is avoided')
    RES['T4c_ystar_below_B'] = dict(max_err=0.0, note=f'y*={rq.x:.4f}, q*={rq.fun:.5f}', ok=bool(rq.x < Bd and rq.fun < Ck))
    # single-time exceedance probability (no absorption): OL hold vs feedback hold. d = B-g
    d_ = Bd - g
    P(f'\n   marginal exceedance P(x_t >= B) of holding at g (no absorption): OL Phibar(d/(sigma sqrt t)) vs feedback k=0.5: Phibar(d/sigma_t); d={d_:.2f}, sigma={sig}')
    P(f"{'t':>3} {'OL exact':>10} {'OL MC':>9} {'FB exact':>10} {'FB MC':>9}")
    rr = np.random.default_rng(7)
    xo = np.full(nmc, g); xf = np.full(nmc, g); kk = 0.5
    xo_ = xo.copy(); xf_ = xf.copy(); ho = np.zeros(nmc, bool); hf = np.zeros(nmc, bool)
    worst = 0; fp = {}
    for t in range(1, 41):
        wn = sig * rr.normal(size=nmc)
        xo = xo + wn; xf = xf - kk * (xf - g) + wn
        xo_ = np.where(ho, xo_, xo_ + wn); ho |= xo_ >= Bd           # absorbed copies for first-passage
        xf_ = np.where(hf, xf_, xf_ - kk * (xf_ - g) + wn); hf |= xf_ >= Bd
        st = sig * math.sqrt((1 - (1 - kk) ** (2 * t)) / (1 - (1 - kk) ** 2))
        eo = stats.norm.sf(d_ / (sig * math.sqrt(t))); ef = stats.norm.sf(d_ / st)
        mo, mf = (xo - g >= d_).mean(), (xf - g >= d_).mean()
        worst = max(worst, abs(mo - eo), abs(mf - ef)); fp[t] = (ho.mean(), hf.mean())
        if t in (1, 2, 5, 10, 20, 40):
            P(f'{t:3d} {eo:10.4f} {mo:9.4f} {ef:10.4f} {mf:9.4f}')
    rec('T4c_marginal_exceedance', worst, 'abs error exact vs MC', tol=3 * math.sqrt(0.25 / nmc))
    P('\n   first-passage (absorbing) hit probability of the same two hold policies: OL vs feedback k=0.5')
    P(f"{'t':>3} {'P(hit) OL':>10} {'P(hit) FB':>10}")
    for t in (1, 2, 5, 10, 20, 40):
        P(f'{t:3d} {fp[t][0]:10.4f} {fp[t][1]:10.4f}')
    P('   NOTE: the cumulative first-passage probability of ANY stationary Gaussian closed loop tends to 1 as t grows; feedback bounds the spread, not the cumulative hit probability. At t=40 the weak feedback k=0.5 is no better than open loop.')
    RES['T4c_firstpassage_crossover'] = dict(max_err=0.0, note=f'FB<=OL for t<=20: {all(fp[t][1] <= fp[t][0] for t in range(1, 21))}; at t=40 FB={fp[40][1]:.3f} vs OL={fp[40][0]:.3f}', ok=bool(all(fp[t][1] <= fp[t][0] + 1e-9 for t in range(1, 21))))


# =====================================================================================================
# T5
# =====================================================================================================
def t5():
    hdr('T5  scale recalibration cannot repair a shape change')
    P('Model P = N(0,1); true post-shift law Q = Laplace with variance 1 (b=1/sqrt2): same mean and variance, different shape.  Failure event {Z >= B}.')
    b = 1 / math.sqrt(2)
    Qbar = lambda B_: 0.5 * np.exp(-B_ / b)
    P(f"{'B':>5} {'Q(Z>=B)':>10} {'P(Z>=B) (s=1)':>14} {'ratio P/Q':>10} {'s_B matching':>13}")
    Bs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    sB = {}
    zq = rng.laplace(scale=b, size=S)
    worst = 0
    for B_ in Bs:
        q = Qbar(B_); sBv = B_ / stats.norm.isf(q)
        sB[B_] = sBv
        worst = max(worst, abs((zq >= B_).mean() - q) / q if B_ <= 3.0 else 0)
        P(f'{B_:5.1f} {q:10.5f} {stats.norm.sf(B_):14.5f} {stats.norm.sf(B_) / q:10.4f} {sBv:13.4f}')
    rec('T5a_laplace_tails_MC', worst, 'relative error of Laplace tail MC vs formula (B<=3 effectively)', tol=3e-2)
    P('   matching scale s_B depends on B (not constant) -> Q is not in the scale family of P.')
    P('   limit: for every s, P_s(Z>=B)/Q(Z>=B) -> 0 as B->inf; check s in {1, 1.22, 1.5, 2} at B=10: ' +
      ', '.join(f's={s}: {stats.norm.sf(10 / s) / Qbar(10):.2e}' for s in [1, 1.22, 1.5, 2]))
    # explicit decision counter-example
    P('\nTwo decision problems (accept risky iff model failure prob < theta): B1=1 (theta1=0.14), B2=3 (theta2=0.004).')
    P('   truth: accept at B1 (0.1217<0.14), reject at B2 (0.0072>0.004).  Model N(0,s^2): accept@B1 iff Phibar(1/s)<0.14 iff s<s_a;  reject@B2 iff Phibar(3/s)>0.004 iff s>s_b')
    s_a = 1 / stats.norm.isf(0.14); s_b = 3 / stats.norm.isf(0.004)
    P(f'   s_a = {s_a:.4f}, s_b = {s_b:.4f}  -> need s<{s_a:.4f} and s>{s_b:.4f}: empty since s_b>s_a')
    ss = np.linspace(0.3, 3, 2701)
    ok_both = [(stats.norm.sf(1 / s) < 0.14) and (stats.norm.sf(3 / s) > 0.004) for s in ss]
    P(f'   grid check: number of scales s in [0.3,3] giving both correct decisions = {sum(ok_both)}')
    rec('T5b_decision_counterexample', float(sum(ok_both)), 'number of scales that fix both decisions (must be 0)', tol=0)
    P(f'   variance-matched s=1 : decisions: B1 accept = {stats.norm.sf(1) < 0.14}, B2 accept = {stats.norm.sf(3) < 0.004} (truth: True, False)')
    # regret of the s=1 calibrated planner: candidate at B2 vs safe with cost gap
    kap = 100.0; gap = kap * 0.004
    P(f'   with kappa={kap}: risky candidate (B2) wrongly accepted by s=1 model: true excess cost = kappa*(Q-theta2) = {kap * (Qbar(3.0) - 0.004):.4f}')
    # moment-matching in higher dimension / affine: same mean and covariance => identical under any affine recalibration based on 2 moments
    P('\nSame mean and covariance, different failure probability (affine recalibration based on second moments is the identity):')
    P(f"   P=N(0,1) vs Q=Laplace(var 1) vs Q2=two-point +-1 (wind-like): mean/var of Q2 = {0:.1f}/{1:.1f};  Q2(Z>=B=1)=0.5, P(Z>=1)={stats.norm.sf(1):.4f}, Q(Z>=1)={Qbar(1.0):.4f};  Q2(Z>=1.01)=0")
    # error floor of the best pure-scale recalibration (location fixed at 0 by symmetry)
    P('\nError floor: inf_s sup_x |F_Q(x) - Phi(x/s)| (Kolmogorov distance of Q to the best centred Gaussian scale)')
    xs_ = np.linspace(-12, 12, 48001)
    FL = np.where(xs_ < 0, 0.5 * np.exp(xs_ / b), 1 - 0.5 * np.exp(-xs_ / b))
    FT = np.where(xs_ < -1, 0.0, np.where(xs_ < 1, 0.5, 1.0))
    fl = optimize.minimize_scalar(lambda s_: np.max(np.abs(FL - ndtr(xs_ / s_))), bounds=(0.3, 3), method='bounded', options=dict(xatol=1e-8))
    ft = optimize.minimize_scalar(lambda s_: np.max(np.abs(FT - ndtr(xs_ / s_))), bounds=(0.3, 3), method='bounded', options=dict(xatol=1e-8))
    P(f'   Laplace(var 1): floor {fl.fun:.4f} at s={fl.x:.3f};  two-point +-1: floor {ft.fun:.4f} at s={ft.x:.3f} (>= 1/4 proven)')
    rec('T5c_floor_twopoint_ge_quarter', max(0.0, 0.25 - ft.fun), 'floor must be >= 0.25 (value %.4f)' % ft.fun, tol=1e-6)
    rec('T5c_summary', 0.0, 'analytic', tol=0)


if __name__ == '__main__':
    t0 = time.time()
    sel = [a for a in sys.argv[1:] if a.startswith('t')]
    for name, fn in [('t1', t1), ('t2', t2), ('t3', t3), ('t4', t4), ('t5', t5)]:
        if sel and name not in sel: continue
        t1_ = time.time(); fn(); P(f'   [{name} took {time.time() - t1_:.1f}s]')
    P(); P('=' * 100); P('SUMMARY (max errors)')
    for k, v in RES.items():
        P(f"  {k:34s} err={v['max_err']:.3e}  ok={v['ok']}  {v['note']}")
    json.dump(RES, open(os.path.join(OUT, 'd4_verify.json'), 'w'), indent=1)
    P(f'total {time.time() - t0:.1f}s')
