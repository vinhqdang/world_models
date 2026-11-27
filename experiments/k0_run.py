"""K0 pre-kill runner: python k0_run.py <setting: g15|g30|tanh> <seed>. Protocol: results/k0/PROTOCOL.md."""
import warnings; warnings.filterwarnings("ignore")
import sys, math, json, itertools, time, numpy as np, torch
sys.path.insert(0, "/home/user/world_models")
from cwk import kernels as KR
torch.set_default_dtype(torch.float64); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
import os; NST = int(os.environ.get('K0_STEPS', 3000)); setting, seed = sys.argv[1], int(sys.argv[2])
PI = {"g15": 0.15, "g30": 0.30, "tanh": None}[setting]; noise = 0.05
rng = np.random.default_rng(seed); torch.manual_seed(seed)
d, R, n, nt = 2, 3, 80000, 20000
J = np.array([[1.0, 0.3], [-0.2, 0.8]])
def Gtrue(x):
    P = np.zeros((len(x), 2, 3)); P[:, 0, 0] = 1; P[:, 1, 0] = -1; P[:, 0, 1] = x[:, 0]; P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)
def draw(m):
    if PI is None: return np.tanh(2.0 * rng.standard_normal((m, R))) / 0.78
    return (rng.random((m, R)) < PI) * rng.uniform(-1, 1, (m, R)) / math.sqrt(PI / 3)
x = rng.uniform(-1, 1, (n, 2)); xi = draw(n); r = np.einsum("bdr,br->bd", Gtrue(x), xi) + noise * rng.standard_normal((n, d))
xt, xt2, xit = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)), draw(nt)
rt = np.einsum("bdr,br->bd", Gtrue(xt), xit) + noise * rng.standard_normal((nt, d)); tgt = np.einsum("bdr,br->bd", Gtrue(xt2), xit)
SST = ((tgt - tgt.mean(0)) ** 2).sum(1)
sqerr = {}
def R2(name, p):
    sqerr[name] = ((p - tgt) ** 2).sum(1); return float(100 * (1 - sqerr[name].sum() / SST.sum()))
psi = lambda z: np.c_[np.ones(len(z)), z]
perm = np.random.default_rng(1000 + seed).permutation(n); va, tr = perm[: n // 10], perm[n // 10:]
res = {"setting": setting, "seed": seed}; t00 = time.time()
Gs_t, Gt_t = Gtrue(xt), Gtrue(xt2)
def blup(Gs, Gt, nug):
    A = Gs @ Gs.transpose(0, 2, 1) + nug * np.eye(d)
    return np.einsum("bdr,br->bd", Gt, np.einsum("bdr,bd->br", Gs, np.linalg.solve(A, rt[..., None])[..., 0]))
res["oracle_blup"] = R2("oracle_blup", blup(Gs_t, Gt_t, noise ** 2))
S_ = KR.hh_moments(psi(x[tr]), r[tr]); G0, _ = KR.bm_lift(S_, 3, d, R, restarts=8, seed=seed)
Gh = lambda z: np.einsum("bq,qdr->bdr", psi(z), G0)
res["hh_blup"] = R2("hh_blup", blup(Gh(xt), Gh(xt2), max(noise ** 2, 1e-3)))
print(setting, seed, "oracle", res["oracle_blup"], "hh", res["hh_blup"], flush=True)
Ptr, rtr = torch.tensor(psi(x[tr])), torch.tensor(r[tr]); Pva, rva = torch.tensor(psi(x[va])), torch.tensor(r[va])
Pt, Pt2, rtT = torch.tensor(psi(xt)), torch.tensor(psi(xt2)), torch.tensor(rt)
def inv2(c):   # closed-form inverse/logdet of (...,2,2) SPD
    a, b, dd = c[..., 0, 0], c[..., 0, 1], c[..., 1, 1]; det = a * dd - b * b
    return a, b, dd, det
def quad_solve(c, v):   # c (...,2,2), v (...,2) -> c^{-1} v, logdet
    a, b, dd, det = inv2(c)
    s0 = (dd * v[..., 0] - b * v[..., 1]) / det; s1 = (-b * v[..., 0] + a * v[..., 1]) / det
    return torch.stack([s0, s1], -1), torch.log(det)
def batched(fn, P, rr, bs=2000):
    outs = [fn(P[i:i + bs], rr[i:i + bs]) for i in range(0, len(P), bs)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
# ---------------- IDT ----------------
Kc = 4; combos = torch.tensor(list(itertools.product(range(Kc), repeat=R))); ar = torch.arange(R)
class IDT:
    def __init__(s, g):
        s.Gq = torch.tensor(G0).clone().requires_grad_(True)
        s.m = (torch.linspace(-1.2, 1.2, Kc).repeat(R, 1) + 0.1 * torch.randn(R, Kc, generator=g)).requires_grad_(True)
        s.ls = torch.full((R, Kc), -0.5).requires_grad_(True); s.lw = torch.zeros(R, Kc).requires_grad_(True)
        s.logn = torch.full((d,), math.log(0.01)).requires_grad_(True)
    def params(s): return [s.Gq, s.m, s.ls, s.lw, s.logn]
    def src(s):
        w = torch.softmax(s.lw, -1); sd = torch.exp(s.ls); mu = (w * s.m).sum(-1, keepdim=True)
        var = (w * (sd ** 2 + (s.m - mu) ** 2)).sum(-1, keepdim=True); z = torch.sqrt(var)
        return torch.log_softmax(s.lw, -1), (s.m - mu) / z, sd / z
    def fwd(s, P, rr):
        lwr, m, sd = s.src(); G = torch.einsum("bq,qdr->bdr", P, s.Gq)
        lwc = lwr[ar, combos].sum(-1); mc = m[ar, combos]; sc = sd[ar, combos] ** 2          # (C,), (C,R), (C,R)
        Gs = G[:, None] * sc[None, :, None, :]; nvec = 1e-3 + torch.exp(s.logn)
        cov = Gs @ G[:, None].transpose(-1, -2) + torch.diag(nvec)
        dev = rr[:, None] - torch.einsum("bdr,cr->bcd", G, mc); sol, ld = quad_solve(cov, dev)
        lp = lwc[None] - 0.5 * (dev * sol).sum(-1) - 0.5 * ld
        xi_c = mc[None] + torch.einsum("bcdr,bcd->bcr", Gs, sol)
        return torch.logsumexp(lp, 1) - d * 0.5 * math.log(2 * math.pi), (torch.softmax(lp, 1)[..., None] * xi_c).sum(1)
def heldout(model):
    with torch.no_grad(): return float(-batched(model.fwd, Pva, rva)[0].mean())
def fit_idt(rs):
    g = torch.Generator().manual_seed(seed * 100 + rs); mdl = IDT(g); ps = mdl.params()
    opt = torch.optim.Adam(ps, lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NST); best, bs = 1e9, None
    rg = np.random.default_rng(seed * 100 + rs)
    for st in range(NST):
        i = torch.as_tensor(rg.choice(len(tr), 512)); l = -mdl.fwd(Ptr[i], rtr[i])[0].mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(ps, 5.0); opt.step(); sch.step()
        if (st + 1) % max(1, NST // 6) == 0:
            v = heldout(mdl)
            if v < best: best, bs = v, [p.detach().clone() for p in ps]
    with torch.no_grad():
        for p, b in zip(ps, bs): p.copy_(b)
    return mdl, best
t0 = time.time(); fits = [fit_idt(rs) for rs in range(4)]; res["idt_val_nll_restarts"] = [f[1] for f in fits]
idt, vnll = min(fits, key=lambda f: f[1]); res["idt_val_nll"] = vnll
with torch.no_grad():
    xh = batched(idt.fwd, Pt, rtT)[1]; Gi = lambda P: torch.einsum("bq,qdr->bdr", P, idt.Gq)
    res["idt"] = R2("idt", torch.einsum("bdr,br->bd", Gi(Pt2), xh).numpy())
print(setting, seed, "idt", res["idt"], "val", vnll, f"{time.time()-t0:.0f}s", flush=True)
# ---------------- rc_eb_gmm ----------------
Kg = 32
class GMM:
    def __init__(s):
        s.m = torch.randn(Kg, R).requires_grad_(True); s.Lr = torch.zeros(Kg, R, R).requires_grad_(True)
        s.ld = torch.full((Kg, R), -1.0).requires_grad_(True); s.lw = torch.zeros(Kg).requires_grad_(True); s.logn = torch.tensor(math.log(0.02)).requires_grad_(True)
    def params(s): return [s.m, s.Lr, s.ld, s.lw, s.logn]
    def fwd(s, P, rr):
        G = torch.einsum("bq,qdr->bdr", P, torch.tensor(G0))
        L = torch.tril(s.Lr, -1) + torch.diag_embed(torch.exp(s.ld)); S = L @ L.transpose(-1, -2)
        GS = torch.einsum("bdr,crs->bcds", G, S); cov = GS @ G[:, None].transpose(-1, -2) + (1e-4 + torch.exp(s.logn)) * torch.eye(d)
        dev = rr[:, None] - torch.einsum("bdr,cr->bcd", G, s.m); sol, ld = quad_solve(cov, dev)
        lp = torch.log_softmax(s.lw, 0)[None] - 0.5 * (dev * sol).sum(-1) - 0.5 * ld
        return torch.logsumexp(lp, 1) - d * 0.5 * math.log(2 * math.pi), (torch.softmax(lp, 1)[..., None] * (s.m[None] + torch.einsum("bcds,bcd->bcs", GS, sol))).sum(1)
def fit_gmm(rs):
    torch.manual_seed(seed * 100 + 50 + rs); mdl = GMM(); ps = mdl.params(); opt = torch.optim.Adam(ps, lr=1e-2); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, NST)
    rg = np.random.default_rng(seed * 100 + 50 + rs)
    for st in range(NST):
        i = torch.as_tensor(rg.choice(len(tr), 1024)); l = -mdl.fwd(Ptr[i], rtr[i])[0].mean(); opt.zero_grad(); l.backward(); opt.step(); sch.step()
    return mdl, heldout(mdl)
t0 = time.time(); gf = [fit_gmm(rs) for rs in range(4)]; res["gmm_val_nll_restarts"] = [f[1] for f in gf]
gm, gv = min(gf, key=lambda f: f[1]); res["gmm_val_nll"] = gv
with torch.no_grad():
    xh = batched(gm.fwd, Pt, rtT)[1]; res["rc_eb_gmm"] = R2("rc_eb_gmm", torch.einsum("bdr,br->bd", torch.einsum("bq,qdr->bdr", Pt2, torch.tensor(G0)), xh).numpy())
print(setting, seed, "gmm", res["rc_eb_gmm"], "val", gv, f"{time.time()-t0:.0f}s", flush=True)
# ---------------- glr_iso with learned idt columns ----------------
pats = [S for k in range(R + 1) for S in itertools.combinations(range(R), k)]
with torch.no_grad():
    lwr, ms, sds = idt.src(); nd = (1e-3 + torch.exp(idt.logn)).numpy(); nsd = float(np.sqrt(nd.mean()))
    # amplitude box from the learned source laws (0.1% / 99.9% quantiles on a grid)
    box = []
    for j in range(R):
        zz = np.linspace(-8, 8, 40001); w = torch.exp(lwr[j]).numpy(); dens = sum(w[k] * np.exp(-0.5 * ((zz - ms[j, k].item()) / sds[j, k].item()) ** 2) / sds[j, k].item() for k in range(Kc))
        cdf = np.cumsum(dens); cdf /= cdf[-1]; box.append((zz[np.searchsorted(cdf, 0.001)], zz[np.searchsorted(cdf, 0.999)]))
    box = np.array(box)
    Gi_np = lambda P: torch.einsum("bq,qdr->bdr", P, idt.Gq).detach().numpy()
def glr_fit(Gs, rr, lam):
    cost = np.empty((len(rr), len(pats))); U = np.zeros((len(rr), len(pats), R)); rss = np.empty_like(cost)
    for pi_, S in enumerate(pats):
        if len(S) == 0: rss[:, pi_] = (rr ** 2).sum(1); cost[:, pi_] = rss[:, pi_] / nsd ** 2; continue
        Gk = Gs[:, :, list(S)]; M = Gk.transpose(0, 2, 1) @ Gk + 1e-9 * np.eye(len(S))
        uk = np.linalg.solve(M, np.einsum("bdk,bd->bk", Gk, rr)[..., None])[..., 0]; uk = np.clip(uk, box[list(S), 0], box[list(S), 1])
        rss[:, pi_] = ((rr - np.einsum("bdk,bk->bd", Gk, uk)) ** 2).sum(1); U[:, pi_, list(S)] = uk; cost[:, pi_] = rss[:, pi_] / nsd ** 2
    return cost, rss, U
Gva = Gi_np(Pva); Gtt = Gi_np(Pt); Gt2 = Gi_np(Pt2); sizes = np.array([len(S) for S in pats]); glr_nll = {}; glr_r2 = {}
for lam in [2, 4, 6, 8, 12]:
    _, rss, _ = glr_fit(Gva, rva.numpy(), lam)
    lpri = -0.5 * lam * sizes; lpri = lpri - np.logaddexp.reduce(lpri)
    glr_nll[lam] = float(-np.logaddexp.reduce(lpri[None] - 0.5 * rss / nsd ** 2 - np.log(2 * np.pi * nsd ** 2), axis=1).mean())
    cost, _, U = glr_fit(Gtt, rt, lam); cost = cost + lam * sizes[None]; bi = cost.argmin(1); xhat = U[np.arange(nt), bi]
    glr_r2[lam] = float(100 * (1 - ((np.einsum("bdr,br->bd", Gt2, xhat) - tgt) ** 2).sum() / SST.sum()))
    if lam == 2 or glr_nll[lam] < min(glr_nll.values()): pass
lam_best = min(glr_nll, key=glr_nll.get); res["glr_nll"] = glr_nll; res["glr_r2_by_lambda"] = glr_r2; res["glr_lambda"] = lam_best; res["glr_iso"] = glr_r2[lam_best]
cost, _, U = glr_fit(Gtt, rt, lam_best); cost = cost + lam_best * sizes[None]; xhat = U[np.arange(nt), cost.argmin(1)]; R2("glr_iso", np.einsum("bdr,br->bd", Gt2, xhat))
print(setting, seed, "glr", res["glr_iso"], "lambda", lam_best, flush=True)
# ---------------- exact Bayes by quadrature ----------------
def ndist_cdf(z): return 0.5 * (1 + torch.erf(z / math.sqrt(2)))
def bayes_gated(A, h):
    lev, post = [], []
    Gs = torch.tensor(Gs_t); rtt = torch.tensor(rt)
    for S in pats:
        k = len(S); lp = k * math.log(PI) + (R - k) * math.log(1 - PI)
        if k == 0: lev.append(torch.full((nt,), lp) - 0.5 * (rtt ** 2).sum(1) / noise ** 2); post.append(torch.zeros(nt, R)); continue
        analytic = 0 in S; gridc = [j for j in S if not (analytic and j == 0)]
        ng = int(math.ceil(2 * A / h)); g1 = (torch.arange(ng) + 0.5) / ng * 2 * A - A; hh = 2 * A / ng
        U = torch.cartesian_prod(*([g1] * len(gridc))).reshape(-1, len(gridc)) if gridc else torch.zeros(1, 0)
        Lk = torch.empty(nt); E = torch.zeros(nt, R)
        bs = max(1, int(2e7 // (len(U) * 2)))
        for i in range(0, nt, bs):
            Gb = Gs[i:i + bs]; rb = rtt[i:i + bs]
            e = rb[:, None, :] - (torch.einsum("bdk,mk->bmd", Gb[:, :, gridc], U) if gridc else 0)
            if analytic:
                g = Gb[:, :, 0]; gn2 = (g ** 2).sum(-1)[:, None]; mm = (e * g[:, None]).sum(-1) / gn2; s = noise / torch.sqrt(gn2)
                a, b = (-A - mm) / s, (A - mm) / s; Z = (ndist_cdf(b) - ndist_cdf(a)).clamp_min(1e-300)
                ll = -0.5 * ((e ** 2).sum(-1) - mm ** 2 * gn2) / noise ** 2 + torch.log(Z) + torch.log(s * math.sqrt(2 * math.pi))
                phi = lambda z: torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
                u0 = mm + s * (phi(a) - phi(b)) / Z
            else: ll = -0.5 * (e ** 2).sum(-1) / noise ** 2
            mx = ll.max(1, keepdim=True)[0]; w = torch.exp(ll - mx); sw = w.sum(1)
            # evidence: sum over grid of integral * cell volume * prior density (1/2A)^k  (analytic dim already integrated)
            Lk[i:i + bs] = mx[:, 0] + torch.log(sw) + (len(gridc)) * math.log(hh) - k * math.log(2 * A)
            for jj, c in enumerate(gridc): E[i:i + bs, c] = (w * U[None, :, jj]).sum(1) / sw
            if analytic: E[i:i + bs, 0] = (w * u0).sum(1) / sw
        lev.append(lp + Lk); post.append(E)
    lev = torch.stack(lev, 1); W = torch.softmax(lev, 1); xb = (W[..., None] * torch.stack(post, 1)).sum(1)
    return torch.einsum("bdr,br->bd", torch.tensor(Gt_t), xb).numpy()
def bayes_tanh(npts, m=nt):
    u = (torch.arange(npts) + 0.5) / npts * 2 * (1 / 0.78) - 1 / 0.78; hh = 2 / 0.78 / npts
    z = torch.atanh((0.78 * u).clamp(-1 + 1e-12, 1 - 1e-12)) / 2; dens = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) * 0.78 / (2 * (1 - (0.78 * u) ** 2))
    lw1 = torch.log(dens * hh); U = torch.cartesian_prod(u, u, u)
    lwf = (lw1[:, None, None] + lw1[None, :, None] + lw1[None, None, :]).reshape(-1)
    Gs = torch.tensor(Gs_t[:m]); rtt = torch.tensor(rt[:m]); out = torch.empty(m, R); bs = max(1, int(3e7 // (len(U) * 2)))
    for i in range(0, m, bs):
        e = rtt[i:i + bs, None, :] - torch.einsum("bdk,mk->bmd", Gs[i:i + bs], U)
        ll = -0.5 * (e ** 2).sum(-1) / noise ** 2 + lwf[None]; w = torch.softmax(ll, 1); out[i:i + bs] = w @ U
    return torch.einsum("bdr,br->bd", torch.tensor(Gt_t[:m]), out).numpy()
t0 = time.time()
if PI is not None:
    A = 1 / math.sqrt(PI / 3); pb = bayes_gated(A, 0.03); res["exact_bayes"] = R2("exact_bayes", pb)
    res["exact_bayes_coarse_h0.06"] = float(100 * (1 - ((bayes_gated(A, 0.06) - tgt) ** 2).sum() / SST.sum()))
else:
    pb = bayes_tanh(64); res["exact_bayes"] = R2("exact_bayes", pb)
    r2s = lambda p, m: float(100 * (1 - ((p - tgt[:m]) ** 2).sum() / ((tgt[:m] - tgt[:m].mean(0)) ** 2).sum()))
    res["exact_bayes_res_check_first1000"] = {str(k): r2s(bayes_tanh(k, 1000), 1000) for k in (32, 64, 96)}
    res["exact_bayes_ref"] = "tanh 3-D midpoint quadrature, 64 pts/dim"
print(setting, seed, "bayes", res["exact_bayes"], f"{time.time()-t0:.0f}s", flush=True)
res["total_s"] = time.time() - t00
json.dump(res, open(f"/home/user/world_models/results/k0/raw/{setting}_s{seed}.json", "w"), indent=1)
np.savez_compressed(f"/home/user/world_models/results/k0/raw/{setting}_s{seed}_sq.npz", SST=SST, **sqerr)
