"""EXPLORATORY numeric check of the tangent-quantile identity and its gain over BLUP (scalar outcome, offset + gain drivers).
Y(x) = xi0 + xi1 x. Claim: E[Y(x') | Y(x) = y] = y + (x' - x) * dQ_tau(x)/dx at tau = F(y | x)."""
import numpy as np
rng = np.random.default_rng(0)
def draw(n, kind):
    if kind == "sparse": return (rng.random((n, 2)) < 0.2) * rng.uniform(-1, 1, (n, 2)) / np.sqrt(0.2 / 3)
    if kind == "tanh": return np.tanh(2 * rng.standard_normal((n, 2))) / 0.78
    if kind == "dep": # dependent, non-Gaussian coefficients (no independence): xi1 = sign(xi0) * |z|
        z = rng.standard_normal((n, 2)); return np.c_[z[:, 0], np.sign(z[:, 0]) * np.abs(z[:, 1])]
    return rng.standard_normal((n, 2))
for kind in ("gauss", "tanh", "sparse", "dep"):
    n = 400000; x = rng.uniform(0.2, 1.0, n); xi = draw(n, kind); y = xi[:, 0] + xi[:, 1] * x + 0.0 * rng.standard_normal(n)
    # factual-only estimate of quantile slope: local quantiles in x-bins, finite differences
    nb, nq = 16, 200; edges = np.linspace(0.2, 1.0, nb + 1); xc = 0.5 * (edges[1:] + edges[:-1]); taus = (np.arange(nq) + 0.5) / nq
    Q = np.stack([np.quantile(y[(x >= edges[b]) & (x < edges[b + 1])], taus) for b in range(nb)])        # (nb, nq)
    from numpy.polynomial import polynomial as Pn
    # smooth each quantile curve in x with a cubic fit, differentiate
    co = np.stack([np.polyfit(xc, Q[:, k], 3) for k in range(nq)]); 
    m = 100000; xt = rng.uniform(0.3, 0.9, m); xp = rng.uniform(0.2, 1.0, m); xit = draw(m, kind); yt = xit[:, 0] + xit[:, 1] * xt; tgt = xit[:, 0] + xit[:, 1] * xp
    Qt = np.stack([np.polyval(co[k], xt) for k in range(nq)], 1); dQ = np.stack([np.polyval(np.polyder(co[k]), xt) for k in range(nq)], 1)
    k = np.clip((Qt < yt[:, None]).sum(1), 0, nq - 1); slope = dQ[np.arange(m), k]
    pred_tan = yt + (xp - xt) * slope
    # BLUP with true moments (unit variance independent or as simulated): Cov(Y(x'),Y(x)) / Var(Y(x))
    c00, c01, c11 = np.cov(xi.T)[0, 0], np.cov(xi.T)[0, 1], np.cov(xi.T)[1, 1]
    L = (c00 + c01 * (xt + xp) + c11 * xt * xp) / (c00 + 2 * c01 * xt + c11 * xt ** 2); pred_lin = L * yt
    # rank-preserving (quantile curve) transport
    Qp = np.stack([np.polyval(co[kk], xp) for kk in range(nq)], 1); pred_rank = Qp[np.arange(m), k]
    r2 = lambda p: round(100 * (1 - ((p - tgt) ** 2).mean() / tgt.var()), 1)
    print(kind, "| oracle BLUP", r2(pred_lin), "| tangent-quantile (factual only)", r2(pred_tan), "| rank-preserving", r2(pred_rank), "| residual reuse", r2(yt))
