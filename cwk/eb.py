"""Empirical-Bayes transport on a random-coefficient model r = G(psi) xi + n (same-disturbance transport).

Every arm models the factual residual r (B, d) of one transition as G xi + n with G = G(psi(a)) (B, d, R) given (the
Hildreth-Houck lift or a learned head) and a prior on the shared coefficient xi in R^R, then transports with the
posterior mean: y_hat(a') = G(a') E[xi | r, a].  Arms in this module:

    rc_eb_gmm   full-covariance Gaussian-mixture prior fitted by marginal likelihood (K in {16, 32}, restarts, held-out NLL)
    rc_flow     RealNVP prior, trained by importance-sampled marginal likelihood, posterior mean by self-normalised IS
    rc_npmle    nonparametric (grid-atom) prior fitted by EM, posterior mean over atoms
    idt         independent-factor prior (K location-mixture sources per driver), learned columns
    glr_iso     best-subset (generalised-likelihood-ratio) fault isolation with the idt columns
    quadrature  exact posterior mean on the synthetic syn2 generator (K0 reference)

The code of rc_eb_gmm, idt, glr_iso and the quadrature is a port of experiments/k0_run.py (batched Cholesky replaces the
closed-form 2 x 2 inverses; identical mathematics, d <= 3 and R <= 5 are used). All tensors are float64.
"""
import itertools
import math

import numpy as np
import torch

LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------------------------------------ linear algebra
def chol_solve(C, v):
    """C (..., d, d) SPD, v (..., d) -> (C^{-1} v, logdet C) by batched Cholesky."""
    L = torch.linalg.cholesky(C)
    sol = torch.cholesky_solve(v[..., None], L)[..., 0]
    return sol, 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)


def batched(fn, *arrs, bs=2000):
    """Apply fn(*chunks) -> (a, b) over chunks of the leading axis and concatenate."""
    outs = [fn(*[a[i:i + bs] for a in arrs]) for i in range(0, len(arrs[0]), bs)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def hh_lift(psi, r, R, seed=0, restarts=8):
    """Context-free Hildreth-Houck lift: Burer-Monteiro factors G0 (q, d, R) of the moment estimate of E[r r^T | psi]."""
    from cwk import kernels as KR
    S = KR.hh_moments(psi, r)
    G0, loss = KR.bm_lift(S, psi.shape[1], r.shape[1], R, restarts=restarts, seed=seed)
    return G0


def G_of(psi, G0):
    """psi (B, q) numpy/torch, G0 (q, d, R) -> (B, d, R) float64 torch."""
    return torch.einsum("bq,qdr->bdr", torch.as_tensor(psi, dtype=torch.float64), torch.as_tensor(G0, dtype=torch.float64))


# ------------------------------------------------------------------------------------------------ GMM prior
class GMM:
    """xi ~ sum_c w_c N(m_c, S_c), r | xi ~ N(G xi, n I); returns log p(r) and E[xi | r] in closed form."""

    def __init__(self, K, R, d):
        self.K, self.R, self.d = K, R, d
        f64 = torch.float64
        self.m = torch.randn(K, R, dtype=f64).requires_grad_(True)
        self.Lr = torch.zeros(K, R, R, dtype=f64).requires_grad_(True)
        self.ld = torch.full((K, R), -1.0, dtype=f64).requires_grad_(True)
        self.lw = torch.zeros(K, dtype=f64).requires_grad_(True)
        self.logn = torch.tensor(math.log(0.02), dtype=f64).requires_grad_(True)

    def params(self):
        return [self.m, self.Lr, self.ld, self.lw, self.logn]

    def fwd(self, G, r):
        d = self.d
        L = torch.tril(self.Lr, -1) + torch.diag_embed(torch.exp(self.ld))
        S = L @ L.transpose(-1, -2)
        GS = torch.einsum("bdr,crs->bcds", G, S)
        cov = GS @ G[:, None].transpose(-1, -2) + (1e-4 + torch.exp(self.logn)) * torch.eye(d, dtype=torch.float64)
        dev = r[:, None] - torch.einsum("bdr,cr->bcd", G, self.m)
        sol, ld = chol_solve(cov, dev)
        lp = torch.log_softmax(self.lw, 0)[None] - 0.5 * (dev * sol).sum(-1) - 0.5 * ld
        xi = (torch.softmax(lp, 1)[..., None] * (self.m[None] + torch.einsum("bcds,bcd->bcs", GS, sol))).sum(1)
        return torch.logsumexp(lp, 1) - d * 0.5 * LOG2PI, xi


def fit_gmm(Gtr, rtr, Gva, rva, K=32, steps=3000, lr=1e-2, batch=1024, seed=0, tag=50, rs=0):
    """One GMM fit (seeds as in k0_run.fit_gmm: torch.manual_seed(seed * 100 + tag + rs)). Returns (model, val NLL)."""
    d, R = rtr.shape[1], Gtr.shape[2]
    torch.manual_seed(seed * 100 + tag + rs)
    mdl = GMM(K, R, d)
    ps = mdl.params()
    opt = torch.optim.Adam(ps, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rg = np.random.default_rng(seed * 100 + tag + rs)
    n = len(rtr)
    for _ in range(steps):
        i = torch.as_tensor(rg.choice(n, batch))
        l = -mdl.fwd(Gtr[i], rtr[i])[0].mean()
        opt.zero_grad()
        l.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        v = float(-batched(mdl.fwd, Gva, rva)[0].mean())
    return mdl, v


def fit_rc_eb_gmm(Gtr, rtr, Gva, rva, Ks=(16, 32), restarts=4, steps=3000, seed=0, log=None):
    """Best of ``restarts`` fits per K by held-out factual NLL; K chosen jointly (best overall). K = 32 uses the K0 seeds."""
    best, info = None, []
    for K in Ks:
        for rs in range(restarts):
            mdl, v = fit_gmm(Gtr, rtr, Gva, rva, K=K, steps=steps, seed=seed, tag=50 if K == 32 else 1050, rs=rs)
            info.append(dict(K=K, rs=rs, nll=v))
            if log:
                log(f"  gmm K={K} rs={rs} heldout-nll {v:.4f}")
            if best is None or v < best[1]:
                best = (mdl, v, K)
    return best[0], best[1], dict(K=best[2], fits=info)


@torch.no_grad()
def posterior_xi(model, G, r, bs=4096):
    return batched(model.fwd, G, r, bs=bs)[1]


# ------------------------------------------------------------------------------------------------ IDT (independent factors)
class IDT:
    """r = G xi + n with independent sources xi_j, each a K-component Gaussian location-scale mixture re-standardised to
    zero mean and unit variance; learned columns Gq (q, d, R), diagonal noise (floor 1e-3). Posterior by enumeration of
    the K^R component combinations."""

    def __init__(self, G0, K=4, g=None):
        q, d, R = G0.shape
        self.K, self.R, self.d = K, R, d
        self.combos = torch.tensor(list(itertools.product(range(K), repeat=R)))
        self.ar = torch.arange(R)
        self.Gq = torch.as_tensor(G0, dtype=torch.float64).clone().requires_grad_(True)
        self.m = (torch.linspace(-1.2, 1.2, K, dtype=torch.float64).repeat(R, 1) + 0.1 * torch.randn(R, K, generator=g, dtype=torch.float64)).requires_grad_(True)
        self.ls = torch.full((R, K), -0.5, dtype=torch.float64).requires_grad_(True)
        self.lw = torch.zeros(R, K, dtype=torch.float64).requires_grad_(True)
        self.logn = torch.full((d,), math.log(0.01), dtype=torch.float64).requires_grad_(True)

    def params(self):
        return [self.Gq, self.m, self.ls, self.lw, self.logn]

    def src(self):
        w = torch.softmax(self.lw, -1)
        sd = torch.exp(self.ls)
        mu = (w * self.m).sum(-1, keepdim=True)
        var = (w * (sd ** 2 + (self.m - mu) ** 2)).sum(-1, keepdim=True)
        z = torch.sqrt(var)
        return torch.log_softmax(self.lw, -1), (self.m - mu) / z, sd / z

    def G(self, P):
        return torch.einsum("bq,qdr->bdr", P, self.Gq)

    def fwd_G(self, G, rr):
        lwr, m, sd = self.src()
        lwc = lwr[self.ar, self.combos].sum(-1)
        mc = m[self.ar, self.combos]
        sc = sd[self.ar, self.combos] ** 2
        Gs = G[:, None] * sc[None, :, None, :]
        nvec = 1e-3 + torch.exp(self.logn)
        cov = Gs @ G[:, None].transpose(-1, -2) + torch.diag(nvec)
        dev = rr[:, None] - torch.einsum("bdr,cr->bcd", G, mc)
        sol, ld = chol_solve(cov, dev)
        lp = lwc[None] - 0.5 * (dev * sol).sum(-1) - 0.5 * ld
        xi_c = mc[None] + torch.einsum("bcdr,bcd->bcr", Gs, sol)
        return torch.logsumexp(lp, 1) - self.d * 0.5 * LOG2PI, (torch.softmax(lp, 1)[..., None] * xi_c).sum(1)

    def fwd(self, P, rr):
        return self.fwd_G(self.G(P), rr)


def fit_idt(G0, Ptr, rtr, Pva, rva, K=4, steps=3000, restarts=4, seed=0, batch=512, log=None):
    """Best of ``restarts`` fits by held-out NLL (k0_run.fit_idt; checkpoint every steps/6)."""
    fits = []
    for rs in range(restarts):
        g = torch.Generator().manual_seed(seed * 100 + rs)
        mdl = IDT(G0, K, g)
        ps = mdl.params()
        opt = torch.optim.Adam(ps, lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        best, bs = 1e9, None
        rg = np.random.default_rng(seed * 100 + rs)
        for st in range(steps):
            i = torch.as_tensor(rg.choice(len(Ptr), batch))
            l = -mdl.fwd(Ptr[i], rtr[i])[0].mean()
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(ps, 5.0)
            opt.step()
            sch.step()
            if (st + 1) % max(1, steps // 6) == 0:
                with torch.no_grad():
                    v = float(-batched(mdl.fwd, Pva, rva)[0].mean())
                if v < best:
                    best, bs = v, [p.detach().clone() for p in ps]
        with torch.no_grad():
            for p, b in zip(ps, bs):
                p.copy_(b)
        if log:
            log(f"  idt rs={rs} heldout-nll {best:.4f}")
        fits.append((mdl, best))
    mdl, v = min(fits, key=lambda f: f[1])
    return mdl, v, [f[1] for f in fits]


# ------------------------------------------------------------------------------------------------ glr_iso
def glr_prepare(idt):
    """Noise sd and amplitude box (0.1% / 99.9% quantiles of the learned source laws) of an IDT model."""
    with torch.no_grad():
        lwr, ms, sds = idt.src()
        nd = (1e-3 + torch.exp(idt.logn)).numpy()
        nsd = float(np.sqrt(nd.mean()))
        box = []
        for j in range(idt.R):
            zz = np.linspace(-8, 8, 40001)
            w = torch.exp(lwr[j]).numpy()
            dens = sum(w[k] * np.exp(-0.5 * ((zz - ms[j, k].item()) / sds[j, k].item()) ** 2) / sds[j, k].item()
                       for k in range(idt.K))
            cdf = np.cumsum(dens)
            cdf /= cdf[-1]
            box.append((zz[np.searchsorted(cdf, 0.001)], zz[np.searchsorted(cdf, 0.999)]))
    return nsd, np.array(box)


def glr_fit(Gs, rr, nsd, box, R):
    """All 2^R subsets: least-squares amplitudes clipped to the box. Returns (cost, rss, U)."""
    pats = [S for k in range(R + 1) for S in itertools.combinations(range(R), k)]
    cost = np.empty((len(rr), len(pats)))
    U = np.zeros((len(rr), len(pats), R))
    rss = np.empty_like(cost)
    for pi_, S in enumerate(pats):
        if len(S) == 0:
            rss[:, pi_] = (rr ** 2).sum(1)
            cost[:, pi_] = rss[:, pi_] / nsd ** 2
            continue
        Gk = Gs[:, :, list(S)]
        M = Gk.transpose(0, 2, 1) @ Gk + 1e-9 * np.eye(len(S))
        uk = np.linalg.solve(M, np.einsum("bdk,bd->bk", Gk, rr)[..., None])[..., 0]
        uk = np.clip(uk, box[list(S), 0], box[list(S), 1])
        rss[:, pi_] = ((rr - np.einsum("bdk,bk->bd", Gk, uk)) ** 2).sum(1)
        U[:, pi_, list(S)] = uk
        cost[:, pi_] = rss[:, pi_] / nsd ** 2
    return cost, rss, U, pats


def glr_iso(idt, Gva, rva, Gsrc, rsrc, lams=(2, 4, 6, 8, 12)):
    """Fault isolation by best-subset GLR with the learned idt columns. Lambda by held-out factual NLL of the subset
    mixture (prior weight proportional to exp(-lambda |S| / 2)). Returns dict(xhat_by_lambda, lam, nll)."""
    R = idt.R
    nsd, box = glr_prepare(idt)
    Gva, Gsrc = np.asarray(Gva), np.asarray(Gsrc)
    _, rss, _, pats = glr_fit(Gva, np.asarray(rva), nsd, box, R)
    sizes = np.array([len(S) for S in pats])
    d = Gva.shape[1]
    nll = {}
    for lam in lams:
        lpri = -0.5 * lam * sizes
        lpri = lpri - np.logaddexp.reduce(lpri)
        nll[lam] = float(-np.logaddexp.reduce(lpri[None] - 0.5 * rss / nsd ** 2 - 0.5 * d * np.log(2 * np.pi * nsd ** 2), axis=1).mean())
    cost, _, U, _ = glr_fit(Gsrc, np.asarray(rsrc), nsd, box, R)
    xhat = {}
    for lam in lams:
        c = cost + lam * sizes[None]
        xhat[lam] = U[np.arange(len(c)), c.argmin(1)]
    return dict(xhat=xhat, lam=min(nll, key=nll.get), nll=nll)


# ------------------------------------------------------------------------------------------------ RealNVP prior
class Coupling(torch.nn.Module):
    def __init__(self, R, mask, hidden=64):
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float64))
        k = int(self.mask.sum())
        self.net = torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.Tanh(), torch.nn.Linear(hidden, 2 * (R - k))).double()
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.idx_a = torch.where(self.mask > 0)[0]
        self.idx_b = torch.where(self.mask == 0)[0]

    def forward(self, z):
        xa = z[:, self.idx_a]
        st = self.net(xa)
        s, t = torch.tanh(st[:, :len(self.idx_b)]) * 2.0, st[:, len(self.idx_b):]
        out = z.clone()
        out[:, self.idx_b] = z[:, self.idx_b] * torch.exp(s) + t
        return out


class RealNVP(torch.nn.Module):
    """Generative flow xi = f(z), z ~ N(0, I) (identity at initialisation). Only sampling is needed: the posterior is
    computed by self-normalised importance sampling with prior samples."""

    def __init__(self, R, layers=6, hidden=64, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        masks = []
        for l in range(layers):
            if R == 2:
                m = np.array([1.0, 0.0]) if l % 2 == 0 else np.array([0.0, 1.0])
            else:
                m = np.zeros(R)
                m[rng.permutation(R)[: R // 2]] = 1.0
                if l % 2 == 1:
                    m = 1 - m
                if m.sum() in (0, R):
                    m[0] = 1 - m[0]
            masks.append(m)
        self.layers = torch.nn.ModuleList([Coupling(R, m, hidden) for m in masks])
        self.R = R
        self.logn = torch.nn.Parameter(torch.tensor(math.log(0.02), dtype=torch.float64))

    def forward(self, z):
        for l in self.layers:
            z = l(z)
        return z

    def sample(self, S, gen=None):
        return self(torch.randn(S, self.R, dtype=torch.float64, generator=gen))

    def nll_is(self, G, r, xs):
        """-log mean_s N(r; G xi_s, n I) per point with shared samples xs (S, R): (B,)."""
        d = r.shape[1]
        nv = 1e-4 + torch.exp(self.logn)
        mu = torch.einsum("bdr,sr->bsd", G, xs)
        lp = -0.5 * ((r[:, None] - mu) ** 2).sum(-1) / nv - 0.5 * d * torch.log(2 * math.pi * nv)
        return -(torch.logsumexp(lp, 1) - math.log(len(xs)))


def fit_flow(Gtr, rtr, Gva, rva, steps=1500, lr=2e-3, batch=256, S_train=1024, seed=0, layers=6, log=None):
    torch.manual_seed(seed * 100 + 70)
    R = Gtr.shape[2]
    fl = RealNVP(R, layers, 64, seed=seed)
    opt = torch.optim.Adam(fl.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rg = np.random.default_rng(seed * 100 + 70)
    gen = torch.Generator().manual_seed(seed * 100 + 71)
    best, bs = 1e18, None
    gv = torch.Generator().manual_seed(7)
    for st in range(steps):
        i = torch.as_tensor(rg.choice(len(rtr), batch))
        xs = fl.sample(S_train, gen)
        l = fl.nll_is(Gtr[i], rtr[i], xs).mean()
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(fl.parameters(), 5.0)
        opt.step()
        sch.step()
        if (st + 1) % max(1, steps // 6) == 0 or st + 1 == steps:
            with torch.no_grad():
                xv = fl.sample(4096, gv)
                v = float(torch.cat([fl.nll_is(Gva[j:j + 1024], rva[j:j + 1024], xv) for j in range(0, len(rva), 1024)]).mean())
            if log:
                log(f"  flow step {st + 1} train {float(l):.4f} heldout(IS4096) {v:.4f}")
            if v < best:
                best, bs = v, {k: p.detach().clone() for k, p in fl.state_dict().items()}
    fl.load_state_dict(bs)
    return fl, best


@torch.no_grad()
def posterior_is(sampler_xs, G, r, nv, bs=512):
    """Self-normalised importance-sampling posterior mean with prior samples xs (S, R), noise variance nv."""
    d = r.shape[1]
    out = []
    for i in range(0, len(r), bs):
        mu = torch.einsum("bdr,sr->bsd", G[i:i + bs], sampler_xs)
        lp = -0.5 * ((r[i:i + bs, None] - mu) ** 2).sum(-1) / nv
        w = torch.softmax(lp, 1)
        out.append(w @ sampler_xs)
    return torch.cat(out)


def rc_flow_posterior(fl, G, r, S=4096, seed=0):
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        xs = fl.sample(S, gen)
        nv = float(1e-4 + torch.exp(fl.logn))
    return posterior_is(xs, G, r, nv)


# ------------------------------------------------------------------------------------------------ NPMLE
def npmle_atoms(R, M=2000, b=4.0, seed=0):
    """Atom locations: regular grid for R <= 3 (about M points), scrambled Sobol in [-b, b]^R otherwise."""
    if R <= 3:
        n = max(2, int(round(M ** (1.0 / R))))
        ax = np.linspace(-b, b, n)
        return np.stack(np.meshgrid(*([ax] * R), indexing="ij"), -1).reshape(-1, R)
    from scipy.stats import qmc
    u = qmc.Sobol(R, scramble=True, seed=seed).random(M)
    return (2 * u - 1) * b


def npmle_fit(G, r, atoms, nv, iters=150, bs=4096, w0=None):
    """EM for the mixing weights of a discrete prior on ``atoms`` (M, R): r_i | xi ~ N(G_i xi, nv I)."""
    atoms_t = torch.as_tensor(atoms, dtype=torch.float64)
    M = len(atoms)
    w = torch.full((M,), 1.0 / M, dtype=torch.float64) if w0 is None else w0.clone()
    n = len(r)
    # likelihood matrix kept in float32 chunks only if small; recompute otherwise
    def lik(i):
        mu = torch.einsum("bdr,sr->bsd", G[i:i + bs], atoms_t)
        return -0.5 * ((r[i:i + bs, None] - mu) ** 2).sum(-1) / nv
    cache = [lik(i) for i in range(0, n, bs)] if n * M <= 6e7 else None
    for it in range(iters):
        acc = torch.zeros(M, dtype=torch.float64)
        for c, i in enumerate(range(0, n, bs)):
            L = cache[c] if cache is not None else lik(i)
            p = torch.softmax(L + torch.log(w + 1e-300)[None], 1)
            acc += p.sum(0)
        w = acc / n
    return w


def npmle_nll(G, r, atoms, w, nv, bs=2048):
    atoms_t = torch.as_tensor(atoms, dtype=torch.float64)
    d = r.shape[1]
    out = []
    for i in range(0, len(r), bs):
        mu = torch.einsum("bdr,sr->bsd", G[i:i + bs], atoms_t)
        lp = -0.5 * ((r[i:i + bs, None] - mu) ** 2).sum(-1) / nv - 0.5 * d * math.log(2 * math.pi * nv) + torch.log(w + 1e-300)[None]
        out.append(-torch.logsumexp(lp, 1))
    return torch.cat(out)


def rc_npmle(Gtr, rtr, Gva, rva, nv_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1), M=2000, em_n=20000, iters=150, seed=0, log=None):
    """NPMLE prior on grid atoms; noise variance (relative to tr C / d) chosen by held-out factual NLL. Returns dict."""
    R = Gtr.shape[2]
    rg = np.random.default_rng(seed)
    # support box from the min-norm least-squares coefficients (a lower bound of the support when R > d)
    with torch.no_grad():
        xl = torch.linalg.pinv(Gtr[:4000]) @ rtr[:4000, :, None]
        b = float(np.clip(2.0 * np.quantile(np.abs(xl.numpy()), 0.995), 3.0, 8.0))
    atoms = npmle_atoms(R, M, b, seed)
    sub = torch.as_tensor(rg.choice(len(rtr), min(em_n, len(rtr)), replace=False))
    scale = float((Gtr[:2000] ** 2).sum((1, 2)).mean() / rtr.shape[1])
    best = None
    for c in nv_grid:
        nv = c * scale
        w = npmle_fit(Gtr[sub], rtr[sub], atoms, nv, iters=iters)
        v = float(npmle_nll(Gva, rva, atoms, w, nv).mean())
        if log:
            log(f"  npmle nv={c} heldout-nll {v:.4f}")
        if best is None or v < best[0]:
            best = (v, c, nv, w)
    return dict(nll=best[0], c=best[1], nv=best[2], w=best[3], atoms=atoms, b=b)


@torch.no_grad()
def npmle_posterior(fit, G, r, bs=1024):
    atoms = torch.as_tensor(fit["atoms"], dtype=torch.float64)
    out = []
    for i in range(0, len(r), bs):
        mu = torch.einsum("bdr,sr->bsd", G[i:i + bs], atoms)
        lp = -0.5 * ((r[i:i + bs, None] - mu) ** 2).sum(-1) / fit["nv"] + torch.log(fit["w"] + 1e-300)[None]
        out.append(torch.softmax(lp, 1) @ atoms)
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------ quadrature (syn2)
def _ndist_cdf(z):
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def bayes_gated(Gs_t, Gt_t, rt, PI, noise, A, h):
    """Exact posterior-mean transport on the syn2 gated generator xi_j = Bern(PI) U(-1, 1) / sqrt(PI / 3) (k0_run.bayes_gated):
    pattern enumeration over active drivers; midpoint grid of spacing <= h in the free coordinates; the common-offset
    column (driver 0) is integrated analytically (truncated normal moments) when active."""
    nt, d, R = Gs_t.shape
    pats = [S for k in range(R + 1) for S in itertools.combinations(range(R), k)]
    lev, post = [], []
    Gs, rtt = torch.as_tensor(Gs_t), torch.as_tensor(rt)
    for S in pats:
        k = len(S)
        lp = k * math.log(PI) + (R - k) * math.log(1 - PI)
        if k == 0:
            lev.append(torch.full((nt,), lp, dtype=torch.float64) - 0.5 * (rtt ** 2).sum(1) / noise ** 2)
            post.append(torch.zeros(nt, R, dtype=torch.float64))
            continue
        analytic = 0 in S
        gridc = [j for j in S if not (analytic and j == 0)]
        ng = int(math.ceil(2 * A / h))
        g1 = (torch.arange(ng, dtype=torch.float64) + 0.5) / ng * 2 * A - A
        hh = 2 * A / ng
        U = torch.cartesian_prod(*([g1] * len(gridc))).reshape(-1, len(gridc)) if gridc else torch.zeros(1, 0, dtype=torch.float64)
        Lk = torch.empty(nt, dtype=torch.float64)
        E = torch.zeros(nt, R, dtype=torch.float64)
        bs = max(1, int(2e7 // (len(U) * d)))
        for i in range(0, nt, bs):
            Gb, rb = Gs[i:i + bs], rtt[i:i + bs]
            e = rb[:, None, :] - (torch.einsum("bdk,mk->bmd", Gb[:, :, gridc], U) if gridc else 0)
            if analytic:
                g = Gb[:, :, 0]
                gn2 = (g ** 2).sum(-1)[:, None]
                mm = (e * g[:, None]).sum(-1) / gn2
                s = noise / torch.sqrt(gn2)
                a, b = (-A - mm) / s, (A - mm) / s
                Z = (_ndist_cdf(b) - _ndist_cdf(a)).clamp_min(1e-300)
                ll = -0.5 * ((e ** 2).sum(-1) - mm ** 2 * gn2) / noise ** 2 + torch.log(Z) + torch.log(s * math.sqrt(2 * math.pi))
                phi = lambda z: torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
                u0 = mm + s * (phi(a) - phi(b)) / Z
            else:
                ll = -0.5 * (e ** 2).sum(-1) / noise ** 2
            mx = ll.max(1, keepdim=True)[0]
            w = torch.exp(ll - mx)
            sw = w.sum(1)
            Lk[i:i + bs] = mx[:, 0] + torch.log(sw) + len(gridc) * math.log(hh) - k * math.log(2 * A)
            for jj, c in enumerate(gridc):
                E[i:i + bs, c] = (w * U[None, :, jj]).sum(1) / sw
            if analytic:
                E[i:i + bs, 0] = (w * u0).sum(1) / sw
        lev.append(lp + Lk)
        post.append(E)
    lev = torch.stack(lev, 1)
    W = torch.softmax(lev, 1)
    xb = (W[..., None] * torch.stack(post, 1)).sum(1)
    return torch.einsum("bdr,br->bd", torch.as_tensor(Gt_t), xb).numpy()


def bayes_tanh(Gs_t, Gt_t, rt, noise, npts=64, m=None, scale=0.78):
    """Exact posterior mean on the syn2 tanh generator xi_j = tanh(2 N(0,1)) / 0.78 by 3-D midpoint quadrature in driver space."""
    R = Gs_t.shape[2]
    assert R == 3
    m = len(rt) if m is None else m
    u = (torch.arange(npts, dtype=torch.float64) + 0.5) / npts * 2 * (1 / scale) - 1 / scale
    hh = 2 / scale / npts
    z = torch.atanh((scale * u).clamp(-1 + 1e-12, 1 - 1e-12)) / 2
    dens = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) * scale / (2 * (1 - (scale * u) ** 2))
    lw1 = torch.log(dens * hh)
    U = torch.cartesian_prod(u, u, u)
    lwf = (lw1[:, None, None] + lw1[None, :, None] + lw1[None, None, :]).reshape(-1)
    Gs, rtt = torch.as_tensor(Gs_t[:m]), torch.as_tensor(rt[:m])
    out = torch.empty(m, R, dtype=torch.float64)
    bs = max(1, int(3e7 // (len(U) * 2)))
    for i in range(0, m, bs):
        e = rtt[i:i + bs, None, :] - torch.einsum("bdk,mk->bmd", Gs[i:i + bs], U)
        ll = -0.5 * (e ** 2).sum(-1) / noise ** 2 + lwf[None]
        out[i:i + bs] = torch.softmax(ll, 1) @ U
    return torch.einsum("bdr,br->bd", torch.as_tensor(Gt_t[:m]), out).numpy()


# ------------------------------------------------------------------------------------------------ syn2 generator (K0)
SYN2_J = np.array([[1.0, 0.3], [-0.2, 0.8]])


def syn2_G(x, J=SYN2_J):
    """G(x) = J [ (1, -1)^T | (x1, 0)^T | (0, x2)^T ], x in R^{B x 2}: (B, 2, 3)."""
    P = np.zeros((len(x), 2, 3))
    P[:, 0, 0] = 1
    P[:, 1, 0] = -1
    P[:, 0, 1] = x[:, 0]
    P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)


def syn2_generate(setting, seed, n=80000, nt=20000, noise=0.05):
    """The K0 syn2 generator with the exact random-number order of experiments/k0_run.py. Returns a dict."""
    PI = {"g15": 0.15, "g30": 0.30, "tanh": None}[setting]
    R, d = 3, 2
    rng = np.random.default_rng(seed)

    def draw(m):
        if PI is None:
            return np.tanh(2.0 * rng.standard_normal((m, R))) / 0.78
        return (rng.random((m, R)) < PI) * rng.uniform(-1, 1, (m, R)) / math.sqrt(PI / 3)

    x = rng.uniform(-1, 1, (n, 2))
    xi = draw(n)
    r = np.einsum("bdr,br->bd", syn2_G(x), xi) + noise * rng.standard_normal((n, d))
    xt, xt2, xit = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)), draw(nt)
    rt = np.einsum("bdr,br->bd", syn2_G(xt), xit) + noise * rng.standard_normal((nt, d))
    tgt = np.einsum("bdr,br->bd", syn2_G(xt2), xit)
    perm = np.random.default_rng(1000 + seed).permutation(n)
    return dict(PI=PI, x=x, r=r, xt=xt, xt2=xt2, rt=rt, tgt=tgt, va=perm[: n // 10], tr=perm[n // 10:], noise=noise)
