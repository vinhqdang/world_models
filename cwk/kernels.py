"""Kernel heads for cross-world kriging.

Residuals r = y - mu_hat(H, a) (cross-fitted), in coordinates standardised by the coordinate residual sd. A head
provides a factor G(h, a) in R^{d x R} so that K(a, a' | h) = G(h, a) G(h, a')^T.

kinds
  polar  psi-structured G(h, a) = sum_j psi_j(a) G_j(h), psi(a) = (1, a_axes / a_lim); factual quasi-likelihood of r_t
         under N(0, G G^T + diag n) (Woodbury)          -> "polarization-only"
  echo   unstructured G = Phi(h, a), rank 8; quasi-likelihood of consecutive pairs (r_t, r_{t+1}) with
         Cov = [[G_t G_t^T, kappa G_t G_{t+1}^T], [., G_{t+1} G_{t+1}^T]] + diag n               -> "echo-only"
  joint  psi-structured G with the pair likelihood and kappa (shared G)                           -> "joint"
  ls     per-coordinate rank-one affine scale s_c(h, a) = g_c(h) . psi(a) (signed location-scale)
  flow   Cholesky factor L(h, a) of the conditional covariance (heteroscedastic affine flow)

Transport (``krige``): y_hat(a') = mu(a') + G(a') [G(a)^T G(a) + tau I]^{-1} G(a)^T r (nugget N = 0, ridge tau).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from cwk.predictor import mlp

TAU = 1e-3
NUG_FLOOR = 1e-3
LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------------------------------------ data
class KData:
    """Flattened transitions with standardised residuals and pair links."""

    def __init__(self, h, psi, afeat, r, ep, st, sigma, axes, a_lim, T):
        self.h, self.psi, self.afeat, self.r = [torch.as_tensor(x, dtype=torch.float32) for x in (h, psi, afeat, r)]
        self.ep, self.st, self.sigma, self.axes, self.a_lim, self.T = ep, st, sigma, list(axes), a_lim, T
        n = len(ep)
        nxt = np.arange(n) + 1
        nxt[st >= T - 1] = -1
        self.nxt = nxt
        self.n, self.d, self.q, self.dh = n, r.shape[1], psi.shape[1], h.shape[1]


def psi_features(a, axes, a_lim):
    af = np.asarray(a)[..., axes] / a_lim
    return np.concatenate([np.ones(af.shape[:-1] + (1,)), af], -1), af


def prepare(H, a, y, s, ep, st, mu_oof, coords, axes, a_lim, hscaler, T, sigma=None):
    r = (y - mu_oof)[:, coords]
    if sigma is None:
        sigma = r.std(0) + 1e-9
    psi, af = psi_features(a, axes, a_lim)
    h = (H - hscaler.hm) / hscaler.hs
    return KData(h, psi, af, r / sigma, ep, st, sigma, axes, a_lim, T)


# ------------------------------------------------------------------------------------------------ networks
class PolarG(nn.Module):
    kind = "G"

    def __init__(self, dh, q, d, R, width=256, depth=2):
        super().__init__()
        self.q, self.d, self.R = q, d, R
        self.net = mlp(dh, q * d * R, width, depth)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.3)
            self.net[-1].bias.zero_()
            self.net[-1].bias.add_(torch.randn_like(self.net[-1].bias) * 0.3)

    def forward(self, h, psi, af=None):
        Gj = self.net(h).view(-1, self.q, self.d, self.R)
        return torch.einsum("bq,bqdr->bdr", psi, Gj)


class FreeG(nn.Module):
    kind = "G"

    def __init__(self, dh, q, d, R, width=256, depth=3):
        super().__init__()
        self.d, self.R = d, R
        self.net = mlp(dh + q, d * R, width, depth)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.3)

    def forward(self, h, psi, af=None):
        return self.net(torch.cat([h, psi], -1)).view(-1, self.d, self.R)


class LSNet(nn.Module):
    kind = "LS"

    def __init__(self, dh, q, d, width=256, depth=2):
        super().__init__()
        self.q, self.d = q, d
        self.net = mlp(dh, q * d, width, depth)

    def forward(self, h, psi, af=None):          # s_c(a): (B, d)
        return torch.einsum("bq,bqd->bd", psi, self.net(h).view(-1, self.q, self.d))


class FlowNet(nn.Module):
    kind = "FLOW"

    def __init__(self, dh, q, d, width=256, depth=2, floor=0.03):
        super().__init__()
        self.d, self.floor = d, floor
        self.net = mlp(dh + q, d * (d + 1) // 2, width, depth)
        self.register_buffer("idx", torch.tril_indices(d, d))
        diag = (self.idx[0] == self.idx[1])
        self.register_buffer("diag", diag)

    def forward(self, h, psi, af=None):           # L(a): (B, d, d)
        out = self.net(torch.cat([h, psi], -1))
        L = torch.zeros(out.shape[0], self.d, self.d, dtype=out.dtype)
        vals = torch.where(self.diag, nn.functional.softplus(out) + self.floor, out * 0.2)
        L[:, self.idx[0], self.idx[1]] = vals
        return L


# ------------------------------------------------------------------------------------------------ likelihoods
def nll_factor(G, r, n):
    """-log N(r; 0, G G^T + diag n) per sample. G (B,d,R), r (B,d), n (d,)."""
    G, r, n = G.double(), r.double(), n.double()
    Ni = 1.0 / n
    GN = G * Ni[None, :, None]
    A = torch.eye(G.shape[-1], dtype=G.dtype)[None] + G.transpose(1, 2) @ GN
    L = torch.linalg.cholesky(A)
    v = torch.einsum("bdr,bd->br", GN, r)
    sol = torch.cholesky_solve(v[..., None], L)[..., 0]
    quad = (r * r * Ni).sum(-1) - (v * sol).sum(-1)
    logdet = torch.log(n).sum() + 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
    return 0.5 * (quad + logdet + G.shape[1] * LOG2PI)


def nll_pair(G1, G2, r1, r2, n, kappa):
    """-log N([r1; r2]; 0, U C U^T + diag(n, n)), C = [[1, kappa], [kappa, 1]] (x) I_R, per pair."""
    G1, G2, r1, r2, n = G1.double(), G2.double(), r1.double(), r2.double(), n.double()
    kappa = kappa.double()
    Ni = 1.0 / n
    R = G1.shape[-1]
    I = torch.eye(R, dtype=G1.dtype)[None]
    A1 = G1.transpose(1, 2) @ (G1 * Ni[None, :, None])
    A2 = G2.transpose(1, 2) @ (G2 * Ni[None, :, None])
    c = 1.0 / (1 - kappa ** 2)
    M = torch.cat([torch.cat([c * I + A1, -c * kappa * I.expand_as(A1)], -1),
                   torch.cat([-c * kappa * I.expand_as(A1), c * I + A2], -1)], -2)
    w = torch.cat([torch.einsum("bdr,bd->br", G1 * Ni[None, :, None], r1),
                   torch.einsum("bdr,bd->br", G2 * Ni[None, :, None], r2)], -1)
    L = torch.linalg.cholesky(M)
    sol = torch.cholesky_solve(w[..., None], L)[..., 0]
    quad = (r1 * r1 * Ni).sum(-1) + (r2 * r2 * Ni).sum(-1) - (w * sol).sum(-1)
    logdet = 2 * torch.log(n).sum() + 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1) + R * torch.log(
        1 - kappa ** 2)
    return 0.5 * (quad + logdet + 2 * G1.shape[1] * LOG2PI)


def nll_ls(s, r, n):
    var = s.double() ** 2 + n.double()[None]
    return 0.5 * (torch.log(var) + r.double() ** 2 / var + LOG2PI).sum(-1)


def nll_flow(L, r):
    u = torch.linalg.solve_triangular(L.double(), r.double()[..., None], upper=False)[..., 0]
    ld = torch.log(torch.diagonal(L.double(), dim1=1, dim2=2)).sum(-1)
    return 0.5 * ((u * u).sum(-1) + L.shape[1] * LOG2PI) + ld



def ls_multistart(kd, tr, seed, width, depth, trials=6, short=300):
    """The affine signed scale s_c(a) = g_c . psi(a) has a stationary point at g_a = 0 (the odd term of s^2 cannot
    correct a mis-specified constant); several random starts of a short fit select the basin by training NLL."""
    best, best_net = math.inf, None
    n0 = torch.full((kd.d,), 0.05)
    for tr_i in range(trials):
        torch.manual_seed(seed * 1000 + tr_i)
        net = LSNet(kd.dh, kd.q, kd.d, width, depth)
        with torch.no_grad():
            net.net[-1].weight.mul_(3.0)
            net.net[-1].bias.normal_(0, 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        rng = np.random.default_rng(seed * 1000 + tr_i)
        tot = []
        for i in range(short):
            idx = torch.as_tensor(rng.choice(tr, 512))
            loss = nll_ls(net(kd.h[idx], kd.psi[idx]), kd.r[idx], n0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i >= short - 50:
                tot.append(float(loss))
        if np.mean(tot) < best:
            best, best_net = float(np.mean(tot)), net
    return best_net


def polar1_multistart(kd, tr, seed, width, depth, trials=6, short=300):
    """Rank-one polarization head: as for the signed scale, the bi-affine rank-one factor has a stationary point at
    g_a = 0 for sign-changing gains; random starts of a short fit select the basin by training NLL."""
    best, best_net = math.inf, None
    n0 = torch.full((kd.d,), 0.05)
    for tr_i in range(trials):
        torch.manual_seed(seed * 1000 + 17 + tr_i)
        net = PolarG(kd.dh, kd.q, kd.d, 1, width, depth)
        with torch.no_grad():
            net.net[-1].weight.mul_(3.0)
            net.net[-1].bias.normal_(0, 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        rng = np.random.default_rng(seed * 1000 + 17 + tr_i)
        tot = []
        for i in range(short):
            idx = torch.as_tensor(rng.choice(tr, 512))
            loss = nll_factor(net(kd.h[idx], kd.psi[idx]), kd.r[idx], n0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i >= short - 50:
                tot.append(float(loss))
        if np.mean(tot) < best:
            best, best_net = float(np.mean(tot)), net
    return best_net


# ------------------------------------------------------------------------------------------------ head object
class Head:
    def __init__(self, kind, net, log_n, kappa_raw, kd_meta):
        self.kind, self.net, self.log_n, self.kappa_raw, self.meta = kind, net, log_n, kappa_raw, kd_meta

    @property
    def n(self):
        return NUG_FLOOR + torch.exp(self.log_n)

    @property
    def kappa(self):
        return 0.98 * torch.sigmoid(self.kappa_raw)

    @torch.no_grad()
    def G(self, h, psi, af=None, batch=8192):
        outs = [self.net(h[i:i + batch], psi[i:i + batch]) for i in range(0, len(h), batch)]
        return torch.cat(outs)

    @torch.no_grad()
    def nll_single(self, kd, idx):
        """Held-out factual marginal NLL of single residual vectors (nats per sample)."""
        idx = torch.as_tensor(idx)
        out = []
        for i in range(0, len(idx), 8192):
            j = idx[i:i + 8192]
            out.append(self._single(kd.h[j], kd.psi[j], kd.r[j]))
        return torch.cat(out)

    def _single(self, h, psi, r):
        o = self.net(h, psi)
        if self.kind in ("polar", "echo", "joint"):
            return nll_factor(o, r, self.n)
        if self.kind == "ls":
            return nll_ls(o, r, self.n)
        return nll_flow(o, r)


def fit_head(kd, kind, seed, rank=None, steps=4000, lr=2e-3, batch=512, width=256, depth=None, val_frac=0.1,
             threads=2, verbose=False, eval_every=250):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    eps_ids = np.unique(kd.ep)
    val_eps = set(rng.choice(eps_ids, max(1, int(val_frac * len(eps_ids))), replace=False).tolist())
    is_val = np.array([e in val_eps for e in kd.ep])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    d = kd.d
    if kind in ("polar", "joint"):
        R = rank or 4
        net = PolarG(kd.dh, kd.q, d, R, width, depth or 2) if R > 1 else polar1_multistart(kd, tr, seed, width, depth or 2)
    elif kind == "echo":
        R = rank or 8
        net = FreeG(kd.dh, kd.q, d, R, width, depth or 3)
    elif kind == "ls":
        net = ls_multistart(kd, tr, seed, width, depth or 2)
    elif kind == "flow":
        net = FlowNet(kd.dh, kd.q, d, width, depth or 2)
    else:
        raise ValueError(kind)
    log_n = nn.Parameter(torch.full((d,), math.log(0.05)))
    kappa_raw = nn.Parameter(torch.tensor(-0.5))
    head = Head(kind, net, log_n, kappa_raw, dict(d=d))
    params = list(net.parameters()) + ([log_n] if kind in ("polar", "echo", "joint", "ls") else []) + (
        [kappa_raw] if kind in ("echo", "joint") else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    paired = kind in ("echo", "joint")
    tr_pair = tr[kd.nxt[tr] >= 0] if paired else None
    va_pair = va[kd.nxt[va] >= 0] if paired else None

    def pair_loss(idx):
        nx = torch.as_tensor(kd.nxt[idx])
        idx = torch.as_tensor(idx)
        G1, G2 = net(kd.h[idx], kd.psi[idx]), net(kd.h[nx], kd.psi[nx])
        return nll_pair(G1, G2, kd.r[idx], kd.r[nx], head.n, head.kappa).mean()

    def val_loss():
        with torch.no_grad():
            if paired:
                return float(torch.cat([pair_loss(va_pair[i:i + 4096]).reshape(1) for i in range(0, len(va_pair), 4096)]).mean())
            return float(head.nll_single(kd, va).mean())

    best, best_state = math.inf, None
    for step in range(steps):
        if paired:
            idx = rng.choice(tr_pair, batch)
            loss = pair_loss(idx)
        else:
            idx = torch.as_tensor(rng.choice(tr, batch))
            loss = head._single(kd.h[idx], kd.psi[idx], kd.r[idx]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            vl = val_loss()
            if vl < best:
                best = vl
                best_state = ([p.detach().clone() for p in params],)
            if verbose:
                print(kind, step + 1, round(float(loss), 4), round(vl, 4), float(head.kappa), flush=True)
    with torch.no_grad():
        for p, b in zip(params, best_state[0]):
            p.copy_(b)
    head.val_objective = best
    head.val_episodes = val_eps
    head.net.eval()
    head.heldout_marginal_nll = float(head.nll_single(kd, va).mean())
    head.rank = getattr(net, "R", None)
    return head


# ------------------------------------------------------------------------------------------------ transport
@torch.no_grad()
def krige(head, h, psi_src, psi_tgt, r_std, tau=TAU):
    """Standardised residual transported from the executed action to the target action. All torch tensors (B, .)."""
    if head.kind in ("polar", "echo", "joint"):
        Gs, Gt = head.G(h, psi_src).double(), head.G(h, psi_tgt).double()
        R = Gs.shape[-1]
        A = Gs.transpose(1, 2) @ Gs + tau * torch.eye(R, dtype=Gs.dtype)[None]
        w = torch.linalg.solve(A, torch.einsum("bdr,bd->br", Gs, r_std.double())[..., None])[..., 0]
        return torch.einsum("bdr,br->bd", Gt, w).float()
    if head.kind == "ls":
        ss, st = head.net(h, psi_src).double(), head.net(h, psi_tgt).double()
        return (st * ss / (ss * ss + tau) * r_std.double()).float()
    Ls, Lt = head.net(h, psi_src).double(), head.net(h, psi_tgt).double()
    u = torch.linalg.solve_triangular(Ls, r_std.double()[..., None], upper=False)
    return (Lt @ u)[..., 0].float()


@torch.no_grad()
def krige_ls_nugget(head, h, psi_src, psi_tgt, r_std):
    """Signed location-scale rule with the head's fitted nugget in the denominator (BLUP of a scalar kernel with noise)."""
    ss, st = head.net(h, psi_src).double(), head.net(h, psi_tgt).double()
    return (st * ss / (ss * ss + head.n.detach().double()[None]) * r_std.double()).float()


@torch.no_grad()
def krige_unsigned(head, h, psi_src, psi_tgt, r_std, tau=TAU):
    ss, st = head.net(h, psi_src).double(), head.net(h, psi_tgt).double()
    return (st.abs() * ss.abs() / (ss * ss + tau) * r_std.double()).float()


# ------------------------------------------------------------------------------------------------ moment (HH) estimator
def product_basis(psi):
    q = psi.shape[1]
    cols = [(j, k) for j in range(q) for k in range(j, q)]
    return np.stack([psi[:, j] * psi[:, k] for j, k in cols], 1), cols


def hh_moments(psi, r, ridge=1e-6):
    """Hildreth-Houck style moment regression of r r^T on psi_j psi_k: returns dict (j, k) -> symmetric (d, d) S_jk
    with E[r r^T | a] = sum_{j<=k} psi_j psi_k S_jk."""
    Phi, cols = product_basis(psi)
    N, d = r.shape
    iu = np.triu_indices(d)
    Y = (r[:, :, None] * r[:, None, :])[:, iu[0], iu[1]]
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    B = np.linalg.solve(A, Phi.T @ Y)
    out = {}
    for c, (j, k) in enumerate(cols):
        S = np.zeros((d, d))
        S[iu] = B[c]
        S = S + S.T - np.diag(np.diag(S))
        out[(j, k)] = S
    return out


def bm_lift(S, q, d, R, restarts=8, iters=200, seed=0):
    """Burer-Monteiro factors G_j (q, d, R) with G_j G_j^T ~ S_jj and G_j G_k^T + G_k G_j^T ~ S_jk (LBFGS)."""
    torch.manual_seed(seed)
    tgt = {k: torch.as_tensor(v) for k, v in S.items()}
    scale = math.sqrt(max(float(sum(torch.trace(v) for (j, k), v in tgt.items() if j == k)) / d / q, 1e-6))
    best, bestG = math.inf, None
    for _ in range(restarts):
        G = (torch.randn(q, d, R, dtype=torch.float64) * scale / math.sqrt(R)).requires_grad_(True)
        opt = torch.optim.LBFGS([G], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def loss_fn():
            tot = 0.0
            for (j, k), t in tgt.items():
                if j == k:
                    m = G[j] @ G[j].T
                else:
                    m = G[j] @ G[k].T + G[k] @ G[j].T
                tot = tot + ((m - t) ** 2).sum()
            return tot

        def closure():
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            return l

        opt.step(closure)
        l = float(loss_fn())
        if l < best:
            best, bestG = l, G.detach().clone()
    return bestG.numpy(), best
