"""One fit seed of every arm on one cell: mean predictor (cross-fitted), kernel heads, baselines, losses per group."""
import time

import numpy as np
import torch

from cwk import evalpop as EP, kernels as KR, metrics as M, oracle as OR, predictor as P, cvae as CV
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


def _delta_raw(assumed, eps, A, cell):
    """Raw-action perturbation (B, p) of the assumed command channels for eps (B, J) and actions A (B, p)."""
    out = torch.zeros_like(A)
    x = A[:, cell.axes] / cell.a_lim
    dirn = torch.as_tensor(cell.dirn, dtype=A.dtype)
    pert = 0
    for j, ch in enumerate(assumed):
        kw = {k: v for k, v in ch.items() if k not in ("kind", "scale")}
        m = torch.as_tensor(profile(ch.get("kind", "affine"), x.numpy(), **kw), dtype=A.dtype)
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
    cf = P.crossfit(H, a, y, s, ep, seed, epochs=cfg.get("epochs", 30), width=cfg.get("width", 512), batch=cfg.get("batch", 256))
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
    losses, diag = {}, {}

    def record(name, pred):
        losses[name] = {g: ev.loss(pred) for g, ev in evs.items()}

    def from_delta(deltas):
        pred = base.copy()
        for k in range(1, K):
            pred[:, :, k, coords] = mu_pop[:, k][:, None, coords] + sigma * deltas[k - 1].numpy().reshape(N, E, d)
        return pred

    record("trained", base)
    record("rr_learned", EP.residual_reuse(mu_pop, truth))
    heads = {}

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
        else:
            raise ValueError(arm)
        log(f"[{cell.name} s{seed}] arm {arm} done {time.time() - t0:.0f}s "
            f"{ {g: round(1 - losses[arm][g].mean() / evs[g].var, 3) for g in evs} }", flush=True)

    # factual diagnostics: overidentification between polarization-implied and echo-implied cross-kernels
    if "polar" in heads and "echo" in heads:
        diag["overid"] = overidentification(kd, heads["polar"], heads["echo"])
    return dict(losses=losses, diag=diag, seconds=time.time() - t0)


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
