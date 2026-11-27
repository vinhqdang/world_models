"""One fit seed of every arm on one cell: mean predictor (cross-fitted), kernel heads, baselines, losses per group."""
import time

import numpy as np
import torch

from cwk import eb as EB, evalpop as EP, kernels as KR, metrics as M, oracle as OR, predictor as P, cvae as CV
from cwk.analytic import VAR_TANH
from cwk.sim import profile


class Cell:
    """Everything an arm needs to know about a cell (structure knowledge is explicit)."""

    def __init__(self, name, col, pop, coords, groups, axes, a_lim, act_gain, nominal=None, true_cmd=None, dev=None,
                 iid=1.0, primary=None, dirn=None):
        self.name, self.col, self.pop, self.coords, self.groups = name, col, pop, list(coords), groups
        self.axes, self.a_lim, self.act_gain, self.nominal, self.true_cmd = axes, a_lim, act_gain, nominal, true_cmd
        self.dev, self.iid, self.primary = dev, iid, primary or list(groups)[0]
        self.dirn = np.array([(-1.0) ** i for i in range(len(axes))]) if dirn is None else dirn
        self.world, self.r = None, None

    def channel_dirn(self, ch):
        """Direction over the declared axes of a command channel (``axis``: one actuator, ``dir``: explicit)."""
        if "dir" in ch:
            return np.asarray(ch["dir"], float)
        if "axis" in ch:
            v = np.zeros(len(self.axes))
            v[ch["axis"]] = (-1.0) ** ch["axis"]
            return v
        return self.dirn


def _delta_raw(assumed, eps, A, cell):
    """Raw-action perturbation (B, p) of the assumed command channels for eps (B, J) and actions A (B, p)."""
    out = torch.zeros_like(A)
    x = A[:, cell.axes] / cell.a_lim
    pert = 0
    for j, ch in enumerate(assumed):
        kw = {k: v for k, v in ch.items() if k not in ("kind", "scale", "dir", "axis")}
        m = torch.as_tensor(profile(ch.get("kind", "affine"), x.numpy(), **kw), dtype=A.dtype)
        dirn = torch.as_tensor(cell.channel_dirn(ch), dtype=A.dtype)
        pert = pert + ch["scale"] * eps[:, j:j + 1] * dirn * m
    out[:, cell.axes] = pert / cell.act_gain
    return out


def indist_transport(model, cell, assumed, sigma, Hb, sb, A, y0, iters=8, h=1e-2):
    """Input-disturbance abduction by MAP Gauss-Newton on the learned mean, then transport under the same eps."""
    f = model.torch_fn()
    coords = cell.coords
    sig = torch.as_tensor(sigma, dtype=torch.float32)
    B, J = Hb.shape[0], len(assumed)
    eps = torch.zeros(B, J)
    a0 = A[0]
    prior = torch.eye(J) / VAR_TANH
    with torch.no_grad():
        for _ in range(iters):
            base = f(Hb, a0 + _delta_raw(assumed, eps, a0, cell), sb)[:, coords]
            res = (y0 - base) / sig
            Jm = []
            for j in range(J):
                e1, e2 = eps.clone(), eps.clone()
                e1[:, j] += h
                e2[:, j] -= h
                f1 = f(Hb, a0 + _delta_raw(assumed, e1, a0, cell), sb)[:, coords]
                f2 = f(Hb, a0 + _delta_raw(assumed, e2, a0, cell), sb)[:, coords]
                Jm.append((f1 - f2) / (2 * h) / sig)
            Jm = torch.stack(Jm, -1)                       # (B, d, J)
            lhs = Jm.transpose(1, 2) @ Jm + prior
            rhs = (Jm.transpose(1, 2) @ res[..., None])[..., 0] - eps @ prior
            eps = (eps + torch.linalg.solve(lhs, rhs[..., None])[..., 0]).clamp(-1, 1)
        out = []
        for k in range(1, len(A)):
            out.append(f(Hb, A[k] + _delta_raw(assumed, eps, A[k], cell), sb))
    return out


def run_seed(cell, seed, arms, cfg, log=print):
    """Returns dict arm -> group -> per-parent loss (N,), plus diagnostics."""
    torch.set_num_threads(cfg.get("threads", 2))
    t0 = time.time()
    pop, col = cell.pop, cell.col
    truth = np.asarray(pop["truth"])
    N, E, K, D = truth.shape
    coords = cell.coords
    d = len(coords)
    T = col["actions"].shape[1]
    H, a, y, s, ep, st = P.flatten_transitions(col)
    cfk = dict(epochs=cfg.get("epochs", 30), width=cfg.get("width", 512), batch=cfg.get("batch", 256))
    if cfg.get("pred_episodes"):
        cf = crossfit_limited(H, a, y, s, ep, seed, cfg["pred_episodes"], **cfk)
    else:
        cf = P.crossfit(H, a, y, s, ep, seed, **cfk)
    log(f"[{cell.name} s{seed}] predictors {time.time() - t0:.0f}s val {cf['val_loss']:.4f}", flush=True)
    full = cf["full"]
    mu_pop = EP.mean_predictions(full, pop)                                  # (N, K, D)
    sc = full.sc
    kd = KR.prepare(H, a, y, s, ep, st, cf["oof"], coords, cell.axes, cell.a_lim, sc, T)
    sigma = kd.sigma
    Hn, s_pop, A = EP.pop_inputs(pop)
    hs = torch.as_tensor((np.repeat(Hn, E, 0) - sc.hm) / sc.hs, dtype=torch.float32)       # (B, dh)
    psi_pop = [torch.as_tensor(np.repeat(KR.psi_features(A[:, k], cell.axes, cell.a_lim)[0], E, 0), dtype=torch.float32)
               for k in range(K)]
    r_std = torch.as_tensor(((truth[:, :, 0][..., coords] - mu_pop[:, 0][:, None, coords]) / sigma).reshape(N * E, d),
                            dtype=torch.float32)
    evs = {g: M.Evaluator(truth, pop["weights"], idx, n_seeds=cfg.get("n_seeds", 5)) for g, idx in cell.groups.items()}
    base = EP.broadcast(mu_pop, E)
    losses, diag, pk = {}, {}, {}

    def record(name, pred):
        losses[name] = {g: ev.loss(pred) for g, ev in evs.items()}
        if cfg.get("save_pk"):
            pk[name] = {g: M.parent_action_loss(pred, truth, idx).tolist() for g, idx in cell.groups.items()}

    def from_delta(deltas):
        pred = base.copy()
        for k in range(1, K):
            pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + sigma * deltas[k - 1].numpy().reshape(N, E, d)
        return pred

    record("trained", base)
    record("rr_learned", EP.residual_reuse(mu_pop, truth))
    heads = {}
    ctx = None

    def get_head(kind):
        if kind not in heads:
            tt = time.time()
            if kind == "polar1":      # rank-one polarization head (vector signed location-scale)
                heads[kind] = KR.fit_head(kd, "polar", seed, rank=1, steps=cfg.get("steps", 4000))
            else:
                heads[kind] = KR.fit_head(kd, kind, seed, steps=cfg.get("steps", 4000))
            log(f"[{cell.name} s{seed}] head {kind} {time.time() - tt:.0f}s heldout-nll {heads[kind].heldout_marginal_nll:.4f}"
                f" kappa {float(heads[kind].kappa):.3f}", flush=True)
            diag[f"nll_{kind}"] = heads[kind].heldout_marginal_nll
            diag[f"kappa_{kind}"] = float(heads[kind].kappa)
        return heads[kind]

    for arm in arms:
        if arm in ("pck_pol", "pck_echo", "pck_joint", "pck_rank1"):
            kind = {"pck_pol": "polar", "pck_echo": "echo", "pck_joint": "joint", "pck_rank1": "polar1"}[arm]
            hd = get_head(kind)
            record(arm, from_delta([KR.krige(hd, hs, psi_pop[0], psi_pop[k], r_std) for k in range(1, K)]))
        elif arm in ("ls_signed", "ls_unsigned", "ls_signed_nug"):
            hd = get_head("ls")
            fn = {"ls_signed": KR.krige, "ls_unsigned": KR.krige_unsigned, "ls_signed_nug": KR.krige_ls_nugget}[arm]
            record(arm, from_delta([fn(hd, hs, psi_pop[0], psi_pop[k], r_std) for k in range(1, K)]))
        elif arm == "flow":
            hd = get_head("flow")
            record(arm, from_delta([KR.krige(hd, hs, psi_pop[0], psi_pop[k], r_std) for k in range(1, K)]))
        elif arm in ("indist_gain", "indist_plain", "indist_struct"):
            assumed = {"indist_gain": cell.nominal, "indist_plain": [dict(kind="offset", scale=sum(c["scale"] for c in cell.nominal))],
                       "indist_struct": cell.true_cmd}[arm]
            Ab = [torch.as_tensor(np.repeat(A[:, k], E, 0), dtype=torch.float32) for k in range(K)]
            y0 = torch.as_tensor(truth[:, :, 0][..., coords].reshape(N * E, d), dtype=torch.float32)
            sb = torch.as_tensor(np.repeat(s_pop, E, 0), dtype=torch.float32)
            Hb = torch.as_tensor(np.repeat(Hn, E, 0), dtype=torch.float32)
            outs = indist_transport(full, cell, assumed, sigma, Hb, sb, Ab, y0)
            pred = base.copy()
            for k in range(1, K):
                pred[:, :, k] = outs[k - 1].numpy().reshape(N, E, D)
            record(arm, pred)
        elif arm in ("hh_global", "hh_binned"):
            record(arm, from_delta(hh_arm(kd, cell, hs, psi_pop, r_std, K, seed, bins=1 if arm == "hh_global" else cfg.get("bins", 8),
                                          rank=4)))
        elif arm == "cvae":
            ytgt = torch.as_tensor(((y - s)[:, coords] - sc.dm[coords]) / sc.ds[coords], dtype=torch.float32)
            net = CV.fit_cvae(kd.h, kd.psi, ytgt, ep, seed, steps=cfg.get("cvae_steps", 6000))
            y0s = torch.as_tensor(((truth[:, :, 0][..., coords] - s_pop[:, None, coords] - sc.dm[coords]) / sc.ds[coords])
                                  .reshape(N * E, d), dtype=torch.float32)
            pred = base.copy()
            for k in range(1, K):
                dec = CV.cf_transport(net, hs, psi_pop[0], psi_pop[k], y0s).numpy().reshape(N, E, d)
                pred[:, :, k, coords] = s_pop[:, None, coords] + dec * sc.ds[coords] + sc.dm[coords]
            record(arm, pred)
        elif arm == "rho_ref":
            rho, dev_loss = tune_rho(cell, full, kd, cell.dev)
            diag["rho_star"] = rho
            record(arm, base + rho * (EP.residual_reuse(mu_pop, truth) - base))
        elif arm in PAPER_ARMS:
            if ctx is None:
                ctx = make_ctx(cell, seed, cfg, kd, hs, psi_pop, r_std, mu_pop, truth, sigma, full, sc, evs, base, from_delta,
                               record, diag, log, A, E, K, D, d)
            run_paper_arm(ctx, arm)
        else:
            raise ValueError(arm)
        log(f"[{cell.name} s{seed}] arm {arm} done {time.time() - t0:.0f}s "
            f"{ {g: round(1 - losses[arm][g].mean() / evs[g].var, 3) for g in evs} }", flush=True)

    # factual diagnostics: overidentification between polarization-implied and echo-implied cross-kernels
    if "polar" in heads and "echo" in heads:
        diag["overid"] = overidentification(kd, heads["polar"], heads["echo"])
    out = dict(losses=losses, diag=diag, seconds=time.time() - t0)
    if pk:
        out["pk_losses"] = pk
    return out


def tune_rho(cell, full, kd, dev):
    """rho in [0, 1] minimising the primary-group counterfactual loss on a held-out development population
    (counterfactual truth is used: REFERENCE ARM ONLY)."""
    truth = np.asarray(dev["truth"])
    mu = EP.mean_predictions(full, dev)
    rr = EP.residual_reuse(mu, truth)
    base = EP.broadcast(mu, truth.shape[1])
    idx = cell.groups[cell.primary]
    best, brho = np.inf, 0.0
    for rho in np.linspace(0, 1, 101):
        l = M.parent_loss(base + rho * (rr - base), truth, idx).mean()
        if l < best:
            best, brho = l, rho
    return float(brho), best


def hh_arm(kd, cell, hs, psi_pop, r_std, K, seed, bins=8, rank=4):
    """Binned Hildreth-Houck BLUP: moment regression of r r^T on psi_j psi_k within context bins, Burer-Monteiro
    factors per bin, kriging with the bin's factors. No amortisation across bins."""
    h = kd.h.numpy()
    psi = kd.psi.numpy().astype(np.float64)
    r = kd.r.numpy().astype(np.float64)
    q, d = kd.q, kd.d
    if bins == 1:
        lab_tr = np.zeros(len(h), int)
        centers, proj = None, None
    else:
        mean = h.mean(0)
        U, S, Vt = np.linalg.svd(h - mean, full_matrices=False)
        proj = (mean, Vt[:8].T)
        z = (h - mean) @ proj[1]
        rng = np.random.default_rng(seed)
        centers = z[rng.choice(len(z), bins, replace=False)]
        for _ in range(25):
            lab = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            for b in range(bins):
                if (lab == b).sum() > 10:
                    centers[b] = z[lab == b].mean(0)
        lab_tr = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    Gs = []
    for b in range(bins):
        m = lab_tr == b
        S_ = KR.hh_moments(psi[m], r[m])
        G, _ = KR.bm_lift(S_, q, d, rank, restarts=4 if bins > 1 else 8, seed=seed)
        Gs.append(torch.as_tensor(G, dtype=torch.float64))                # (q, d, R)
    hn = hs.numpy()
    if bins == 1:
        lab_te = np.zeros(len(hn), int)
    else:
        zt = (hn - proj[0]) @ proj[1]
        lab_te = ((zt[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    Gb = torch.stack(Gs)[torch.as_tensor(lab_te)]                         # (B, q, d, R)
    out = []
    ps = psi_pop[0].double()
    Gsrc = torch.einsum("bq,bqdr->bdr", ps, Gb)
    R = Gsrc.shape[-1]
    A = Gsrc.transpose(1, 2) @ Gsrc + KR.TAU * torch.eye(R, dtype=torch.float64)[None]
    w = torch.linalg.solve(A, torch.einsum("bdr,bd->br", Gsrc, r_std.double())[..., None])[..., 0]
    for k in range(1, K):
        Gt = torch.einsum("bq,bqdr->bdr", psi_pop[k].double(), Gb)
        out.append(torch.einsum("bdr,br->bd", Gt, w).float())
    return out


@torch.no_grad()
def overidentification(kd, pol, echo, max_pairs=20000):
    """Factual overidentification between the polarization-implied and echo-implied cross-kernels at consecutive
    action pairs of held-out episodes (all computed on the factual data; no counterfactual quantity)."""
    rng = np.random.default_rng(0)
    # both heads were fitted with the same seed and hence the same held-out episodes
    held = np.isin(kd.ep, list(pol.val_episodes & echo.val_episodes))
    idx = np.where((kd.nxt >= 0) & held)[0]
    if len(idx) > max_pairs:
        idx = rng.choice(idx, max_pairs, replace=False)
    nx = kd.nxt[idx]
    i, j = torch.as_tensor(idx), torch.as_tensor(nx)
    Gp1, Gp2 = pol.G(kd.h[i], kd.psi[i]).double(), pol.G(kd.h[i], kd.psi[j]).double()
    Ge1, Ge2 = echo.G(kd.h[i], kd.psi[i]).double(), echo.G(kd.h[i], kd.psi[j]).double()
    Kp = Gp1 @ Gp2.transpose(1, 2)
    Ke = Ge1 @ Ge2.transpose(1, 2)
    kappa = float(echo.kappa)
    rr = kd.r[i].double()[:, :, None] * kd.r[j].double()[:, None, :]
    lp = ((rr - kappa * Kp) ** 2).sum((1, 2))
    le = ((rr - kappa * Ke) ** 2).sum((1, 2))
    rel = float(((Kp - Ke) ** 2).sum() / (Ke ** 2).sum())
    dl = (lp - le).numpy()
    eps_p = kd.ep[idx]
    uniq, inv = np.unique(eps_p, return_inverse=True)
    per = np.bincount(inv, weights=dl) / np.bincount(inv)
    z = float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)) + 1e-12))
    return dict(rel_discrepancy=rel, lag_loss_diff_mean=float(dl.mean()), z=z, n_pairs=int(len(idx)),
                n_episodes=int(len(uniq)), kappa_echo=kappa)


# ================================================================================================ paper arms
PAPER_ARMS = {"hh_binned_nug", "hh_nug_ref", "oracle_kernel_lm", "oracle_quad", "hh_basis_x", "hh_basis_abs", "hh_basis_x_x3",
              "hh_basis_cubic", "glr_iso", "glr_iso_lamo", "rc_eb_gmm", "rc_flow", "rc_npmle", "idt"}
PAPER_PRIVILEGED = {"hh_nug_ref", "oracle_kernel_lm", "oracle_quad", "glr_iso_lamo", "oracle_quad_ex", "oracle_struct_bayes"}
NUG_GRID = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)       # nugget N = c tr(C) / d, C = G G^T
BASES = {"hh_basis_x": "x", "hh_basis_abs": "abs", "hh_basis_x_x3": "x_x3", "hh_basis_cubic": "cubic"}


def crossfit_limited(H, a, y, s, ep, seed, n_ep, **kw):
    """Mean predictor trained on the first ``n_ep`` episodes only (R6): 2-fold cross-fitted residuals inside that subset,
    the single predictor's out-of-sample residuals elsewhere."""
    sub = np.isin(ep, np.unique(ep)[:n_ep])
    cf = P.crossfit(H[sub], a[sub], y[sub], s[sub], ep[sub], seed, **kw)
    oof = np.empty_like(y)
    oof[sub] = cf["oof"]
    oof[~sub] = cf["full"].predict(H[~sub], a[~sub], s[~sub])
    return dict(oof=oof, full=cf["full"], val_loss=cf["val_loss"])


def basis_features(af, kind):
    """Action basis psi(a) from the normalised actuated command x = a / a_lim: (1, x), (1, |x|), (1, x, x^3), (1, x, x^2, x^3)."""
    af = np.asarray(af, np.float64)
    one = np.ones(af.shape[:-1] + (1,))
    parts = {"x": [one, af], "abs": [one, np.abs(af)], "x_x3": [one, af, af ** 3], "cubic": [one, af, af ** 2, af ** 3]}[kind]
    return np.concatenate(parts, -1)


class HHModel:
    """Hildreth-Houck moment estimator with Burer-Monteiro factors, optionally binned on the context (as ``hh_arm``), fitted on
    the transitions ``tr`` only. G(h, psi) -> (B, d, R) float64."""

    def __init__(self, h, psi, r, tr, bins, rank, seed, restarts=None):
        self.bins, self.rank = bins, rank
        q, d = psi.shape[1], r.shape[1]
        ht, pt, rt = h[tr], psi[tr].astype(np.float64), r[tr].astype(np.float64)
        if bins == 1:
            lab, self.proj, self.centers = np.zeros(len(ht), int), None, None
        else:
            mean = ht.mean(0)
            _, _, Vt = np.linalg.svd(ht - mean, full_matrices=False)
            self.proj = (mean, Vt[:8].T)
            z = (ht - mean) @ self.proj[1]
            rng = np.random.default_rng(seed)
            centers = z[rng.choice(len(z), bins, replace=False)]
            for _ in range(25):
                lab = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
                for b in range(bins):
                    if (lab == b).sum() > 10:
                        centers[b] = z[lab == b].mean(0)
            lab = ((z[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            self.centers = centers
        Gs = []
        for b in range(bins):
            m = lab == b
            S_ = KR.hh_moments(pt[m], rt[m])
            G, _ = KR.bm_lift(S_, q, d, rank, restarts=restarts or (4 if bins > 1 else 8), seed=seed)
            Gs.append(torch.as_tensor(G, dtype=torch.float64))
        self.Gs = torch.stack(Gs)                                       # (bins, q, d, R)

    def labels(self, h):
        if self.bins == 1:
            return np.zeros(len(h), int)
        z = (np.asarray(h) - self.proj[0]) @ self.proj[1]
        return ((z[:, None] - self.centers[None]) ** 2).sum(-1).argmin(1)

    def G(self, h, psi):
        lab = torch.as_tensor(self.labels(h))
        return torch.einsum("bq,bqdr->bdr", torch.as_tensor(psi, dtype=torch.float64), self.Gs[lab])


def _nug(G, c):
    d = G.shape[1]
    trC = (G ** 2).sum((1, 2)) / d
    return max(c, 1e-6) * trC + 1e-12


def nug_loglik(G, r, c):
    """Mean factual Gaussian log-likelihood of r under N(0, G G^T + N I), N = c tr(G G^T) / d."""
    d = G.shape[1]
    cov = G @ G.transpose(1, 2) + _nug(G, c)[:, None, None] * torch.eye(d, dtype=G.dtype)
    sol, ld = EB.chol_solve(cov, r)
    return float((-0.5 * ((r * sol).sum(-1) + ld + d * EB.LOG2PI)).mean())


def nug_transport(Gs, Gts, r, c):
    """BLUP with nugget: G' (G^T G + N I)^{-1} G^T r (push-through of K(a', a) (K(a, a) + N I)^{-1} r), N = c tr(C) / d."""
    R = Gs.shape[2]
    A = Gs.transpose(1, 2) @ Gs + _nug(Gs, c)[:, None, None] * torch.eye(R, dtype=Gs.dtype)
    w, _ = EB.chol_solve(A, torch.einsum("bdr,bd->br", Gs, r))
    return [torch.einsum("bdr,br->bd", Gt, w) for Gt in Gts]


def make_ctx(cell, seed, cfg, kd, hs, psi_pop, r_std, mu_pop, truth, sigma, full, sc, evs, base, from_delta, record, diag, log,
             A, E, K, D, d):
    c = dict(cell=cell, seed=seed, cfg=cfg, kd=kd, hs=hs, psi_pop=psi_pop, r_std=r_std, mu_pop=mu_pop, truth=truth, sigma=sigma,
             full=full, sc=sc, evs=evs, base=base, from_delta=from_delta, record=record, diag=diag, log=log, A=A, E=E, K=K,
             D=D, d=d, cache={})
    rng = np.random.default_rng(20_000 + seed)
    eps_ids = np.unique(kd.ep)
    val_e = rng.choice(eps_ids, max(1, int(0.1 * len(eps_ids))), replace=False)
    is_va = np.isin(kd.ep, val_e)
    c["tr"], c["va"] = np.where(~is_va)[0], np.where(is_va)[0]
    c["N"] = truth.shape[0]
    return c


def _hh(ctx, bins, rank, basis="x"):
    key = ("hh", bins, rank, basis)
    if key not in ctx["cache"]:
        kd = ctx["kd"]
        psi = kd.psi.numpy() if basis == "x" else basis_features(kd.afeat.numpy(), basis)
        t0 = time.time()
        m = HHModel(kd.h.numpy(), psi, kd.r.numpy(), ctx["tr"], bins, rank, ctx["seed"])
        ctx["cache"][key] = (m, psi)
        ctx["log"](f"  HH fit bins={bins} rank={rank} basis={basis} {time.time() - t0:.0f}s")
    return ctx["cache"][key]


def _pop_psi(ctx, k, basis):
    if basis == "x":
        return ctx["psi_pop"][k]
    af = KR.psi_features(ctx["A"][:, k], ctx["cell"].axes, ctx["cell"].a_lim)[1]
    return torch.as_tensor(np.repeat(basis_features(af, basis), ctx["E"], 0), dtype=torch.float32)


def _split_G(ctx, model, psi):
    kd, tr, va = ctx["kd"], ctx["tr"], ctx["va"]
    h = kd.h.numpy()
    r = kd.r.double()
    return (model.G(h[tr], psi[tr]), r[tr], model.G(h[va], psi[va]), r[va])


def _dev_inputs(ctx):
    if "dev" not in ctx["cache"]:
        cell, kd, full, sc, sigma = ctx["cell"], ctx["kd"], ctx["full"], ctx["sc"], ctx["sigma"]
        dev = cell.dev
        truth = np.asarray(dev["truth"])
        N, E, K, D = truth.shape
        mu = EP.mean_predictions(full, dev)
        Hn, s_, A = EP.pop_inputs(dev)
        hsd = torch.as_tensor((np.repeat(Hn, E, 0) - sc.hm) / sc.hs, dtype=torch.float32)
        psi = [torch.as_tensor(np.repeat(KR.psi_features(A[:, k], cell.axes, cell.a_lim)[0], E, 0), dtype=torch.float32)
               for k in range(K)]
        coords = cell.coords
        r = torch.as_tensor(((truth[:, :, 0][..., coords] - mu[:, 0][:, None, coords]) / sigma).reshape(N * E, len(coords)),
                            dtype=torch.float32)
        ctx["cache"]["dev"] = dict(truth=truth, mu=mu, hs=hsd, psi=psi, r=r, A=A, E=E, K=K, N=N)
    return ctx["cache"]["dev"]


def _hh_nug_arm(ctx, arm, bins, basis, ref=False):
    cell, kd, d = ctx["cell"], ctx["kd"], ctx["d"]
    model, psi = _hh(ctx, bins, 4, basis)
    Gtr, rtr, Gva, rva = _split_G(ctx, model, psi)
    ll = {c: nug_loglik(Gva, rva, c) for c in NUG_GRID}
    ctx["diag"][f"nug_ll_{arm}"] = {str(c): v for c, v in ll.items()}
    hs = ctx["hs"].numpy()
    Gs = model.G(hs, _pop_psi(ctx, 0, basis).numpy())
    Gts = [model.G(hs, _pop_psi(ctx, k, basis).numpy()) for k in range(1, ctx["K"])]
    r = ctx["r_std"].double()
    if ref:                 # privileged: nugget tuned on the development population's counterfactuals
        dv = _dev_inputs(ctx)
        hd = dv["hs"].numpy()
        Gsd = model.G(hd, dv["psi"][0].numpy())
        Gtd = [model.G(hd, dv["psi"][k].numpy()) for k in range(1, dv["K"])]
        idx = cell.groups[cell.primary]
        sig, N_, E_, D_ = ctx["sigma"], dv["N"], dv["E"], dv["truth"].shape[3]
        best, cbest = np.inf, 0.0
        for c in NUG_GRID:
            dl = nug_transport(Gsd, Gtd, dv["r"].double(), c)
            pred = EP.broadcast(dv["mu"], E_).copy()
            for k in range(1, dv["K"]):
                pred[:, :, k, cell.coords] = dv["mu"][:, k][:, None, cell.coords] + sig * dl[k - 1].numpy().reshape(N_, E_, d)
            l = M.parent_loss(pred, dv["truth"], idx).mean()
            if l < best:
                best, cbest = l, c
        ctx["diag"][f"nug_c_{arm}"] = cbest
        c_use = cbest
    else:
        c_use = max(ll, key=ll.get)
        ctx["diag"][f"nug_c_{arm}"] = c_use
        ctx["diag"][f"nll_{arm}"] = -ll[c_use]
    if basis != "x":
        sub = np.random.default_rng(0).choice(kd.n, min(kd.n, 30000), replace=False)
        Phi, _ = KR.product_basis(psi[sub].astype(np.float64))
        ctx["diag"][f"cond_{arm}"] = float(np.linalg.cond(Phi.T @ Phi))
    ctx["record"](arm, ctx["from_delta"]([t.float() for t in nug_transport(Gs, Gts, r, c_use)]))


def _eb_inputs(ctx):
    if "eb" not in ctx["cache"]:
        cell, cfg = ctx["cell"], ctx["cfg"]
        R = cfg.get("eb_rank") or max(int(cell.r or 3), 2)
        model, psi = _hh(ctx, 1, R)
        G0 = model.Gs[0]
        kd, tr, va = ctx["kd"], ctx["tr"], ctx["va"]
        P_ = torch.as_tensor(psi, dtype=torch.float64)
        r = kd.r.double()
        eb = dict(R=R, G0=G0, Ptr=P_[tr], Pva=P_[va], rtr=r[tr], rva=r[va], Gtr=EB.G_of(psi[tr], G0), Gva=EB.G_of(psi[va], G0),
                  Ps=[ctx["psi_pop"][k].double() for k in range(ctx["K"])], r=ctx["r_std"].double())
        eb["Gs"] = [torch.einsum("bq,qdr->bdr", p, G0) for p in eb["Ps"]]
        ctx["cache"]["eb"] = eb
    return ctx["cache"]["eb"]


def _idt(ctx):
    if "idt" not in ctx["cache"]:
        eb, cfg = _eb_inputs(ctx), ctx["cfg"]
        t0 = time.time()
        m, v, vis = EB.fit_idt(eb["G0"], eb["Ptr"], eb["rtr"], eb["Pva"], eb["rva"], K=4, steps=cfg.get("idt_steps", 3000),
                               restarts=cfg.get("idt_restarts") or (4 if eb["R"] <= 3 else 1),
                               seed=ctx["seed"], log=ctx["log"])
        ctx["diag"]["nll_idt"] = v
        ctx["log"](f"  idt fit {time.time() - t0:.0f}s heldout-nll {v:.4f}")
        ctx["cache"]["idt"] = m
    return ctx["cache"]["idt"]


def _xi_delta(ctx, Gts, xi):
    return [torch.einsum("bdr,br->bd", Gt, xi).float() for Gt in Gts]


def run_paper_arm(ctx, arm):
    cell, cfg, K, d, seed = ctx["cell"], ctx["cfg"], ctx["K"], ctx["d"], ctx["seed"]
    rec, fd, log = ctx["record"], ctx["from_delta"], ctx["log"]
    if arm == "hh_binned_nug":
        _hh_nug_arm(ctx, arm, cfg.get("bins", 8), "x")
    elif arm == "hh_nug_ref":
        _hh_nug_arm(ctx, arm, cfg.get("bins", 8), "x", ref=True)
    elif arm in BASES:
        _hh_nug_arm(ctx, arm, 1, BASES[arm])
    elif arm == "oracle_kernel_lm":
        pred = ctx["base"].copy()
        pred[..., cell.coords] = OR.mc_kriging_lm(ctx["truth"], ctx["mu_pop"], cell.coords)
        rec(arm, pred)
    elif arm == "oracle_quad":
        pred = ctx["base"].copy()
        pred[..., cell.coords] = OR.mc_quadratic(ctx["truth"], cell.coords, ctx["mu_pop"])
        rec(arm, pred)
    elif arm == "rc_eb_gmm":
        eb = _eb_inputs(ctx)
        t0 = time.time()
        mdl, v, info = EB.fit_rc_eb_gmm(eb["Gtr"], eb["rtr"], eb["Gva"], eb["rva"], Ks=cfg.get("eb_Ks", (16, 32)),
                                        restarts=cfg.get("eb_restarts", 4), steps=cfg.get("eb_steps", 3000), seed=seed)
        ctx["diag"]["nll_rc_eb_gmm"], ctx["diag"]["K_rc_eb_gmm"] = v, info["K"]
        log(f"  rc_eb_gmm fit {time.time() - t0:.0f}s K={info['K']} heldout-nll {v:.4f}")
        xi = EB.posterior_xi(mdl, eb["Gs"][0], eb["r"])
        rec(arm, fd(_xi_delta(ctx, eb["Gs"][1:], xi)))
    elif arm == "rc_flow":
        eb = _eb_inputs(ctx)
        t0 = time.time()
        fl, v = EB.fit_flow(eb["Gtr"], eb["rtr"], eb["Gva"], eb["rva"], steps=cfg.get("flow_steps", 1500), seed=seed)
        ctx["diag"]["nll_rc_flow"] = v
        log(f"  rc_flow fit {time.time() - t0:.0f}s heldout-nll(IS) {v:.4f}")
        xi = EB.rc_flow_posterior(fl, eb["Gs"][0], eb["r"], S=cfg.get("flow_S", 4096), seed=seed)
        rec(arm, fd(_xi_delta(ctx, eb["Gs"][1:], xi)))
    elif arm == "rc_npmle":
        eb = _eb_inputs(ctx)
        t0 = time.time()
        fit = EB.rc_npmle(eb["Gtr"], eb["rtr"], eb["Gva"], eb["rva"], M=cfg.get("npmle_M", 2000), iters=cfg.get("npmle_iters", 150),
                          em_n=cfg.get("npmle_n", 20000), seed=seed)
        ctx["diag"]["nll_rc_npmle"], ctx["diag"]["npmle_c"], ctx["diag"]["npmle_box"] = fit["nll"], fit["c"], fit["b"]
        log(f"  rc_npmle fit {time.time() - t0:.0f}s heldout-nll {fit['nll']:.4f} c {fit['c']}")
        xi = EB.npmle_posterior(fit, eb["Gs"][0], eb["r"])
        rec(arm, fd(_xi_delta(ctx, eb["Gs"][1:], xi)))
    elif arm == "idt":
        eb = _eb_inputs(ctx)
        idt = _idt(ctx)
        with torch.no_grad():
            xi = EB.batched(lambda P, r: idt.fwd(P, r), eb["Ps"][0], eb["r"], bs=2000)[1]
            Gts = [idt.G(eb["Ps"][k]) for k in range(1, K)]
        rec(arm, fd(_xi_delta(ctx, Gts, xi)))
    elif arm in ("glr_iso", "glr_iso_lamo"):
        eb = _eb_inputs(ctx)
        idt = _idt(ctx)
        if "glr" not in ctx["cache"]:
            with torch.no_grad():
                Gva = idt.G(eb["Pva"]).numpy()
                Gs0 = idt.G(eb["Ps"][0]).numpy()
                Gts = [idt.G(eb["Ps"][k]).numpy() for k in range(1, K)]
            g = EB.glr_iso(idt, Gva, eb["rva"].numpy(), Gs0, eb["r"].numpy())
            preds = {lam: fd([torch.as_tensor(np.einsum("bdr,br->bd", Gt, xh), dtype=torch.float32) for Gt in Gts])
                     for lam, xh in g["xhat"].items()}
            ctx["cache"]["glr"] = (g, preds)
            ctx["diag"]["glr_lambda"], ctx["diag"]["glr_nll"] = g["lam"], {str(k): v for k, v in g["nll"].items()}
        g, preds = ctx["cache"]["glr"]
        if arm == "glr_iso":
            rec(arm, preds[g["lam"]])
        else:
            prim = ctx["evs"][cell.primary]
            lo = {lam: prim.loss(p).mean() for lam, p in preds.items()}
            lam = min(lo, key=lo.get)
            ctx["diag"]["glr_lambda_oracle"] = lam
            rec(arm, preds[lam])
    else:
        raise ValueError(arm)
