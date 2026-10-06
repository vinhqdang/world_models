"""P1 pilot: action sensitivity of action-conditioned predictors under
state-correlated behaviour actions.

State-based environment with a large action-independent drift, a small
state-dependent action gain, and behaviour data whose actions are a noisy
function of the state (so the state alone nearly determines the action).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DS, DA = 4, 2
ACT_SCALE = 0.3


class Env:
    def __init__(self, seed):
        r = np.random.RandomState(10_000 + seed)
        self.R = torch.tensor(r.randn(DS, DS) * 0.6, dtype=torch.float32)
        G0 = r.randn(DS, DA)
        G0 /= np.linalg.norm(G0, axis=0, keepdims=True)
        self.G0 = torch.tensor(G0, dtype=torch.float32)
        self.Wg = torch.tensor(r.randn(DS, DS * DA) * 0.5, dtype=torch.float32)
        # behaviour feedback gain: pushes the state toward the origin
        self.K = torch.tensor(-1.2 * np.linalg.pinv(G0), dtype=torch.float32) * 0.5

    def gain(self, s):
        return self.G0 + 0.5 * torch.tanh(s @ self.Wg).view(*s.shape[:-1], DS, DA)

    def step(self, s, a, noise=0.0, gen=None):
        drift = 0.9 * s + 0.3 * torch.sin(s @ self.R.T)
        eff = ACT_SCALE * (self.gain(s) @ a.unsqueeze(-1)).squeeze(-1)
        out = drift + eff
        if noise > 0:
            out = out + noise * torch.randn(out.shape, generator=gen)
        return out

    def behaviour_mean(self, s):
        return torch.clamp(s @ self.K.T, -1, 1)

    def collect(self, n, sigma, seed, ep_len=20, wnoise=0.01, uniform=False):
        g = torch.Generator().manual_seed(seed)
        S, A, S2 = [], [], []
        n_ep = n // ep_len
        s = 1.5 * torch.randn(n_ep, DS, generator=g)
        for _ in range(ep_len):
            if uniform:
                a = 2 * torch.rand(n_ep, DA, generator=g) - 1
            else:
                a = torch.clamp(self.behaviour_mean(s) + sigma * torch.randn(n_ep, DA, generator=g), -1, 1)
            s2 = self.step(s, a, wnoise, g)
            S.append(s); A.append(a); S2.append(s2)
            s = s2
        return torch.cat(S), torch.cat(A), torch.cat(S2)


def mlp(i, o, h=128, nl=2):
    L, d = [], i
    for _ in range(nl):
        L += [nn.Linear(d, h), nn.SiLU()]
        d = h
    L.append(nn.Linear(d, o))
    return nn.Sequential(*L)


class Predictor(nn.Module):
    """delta-state predictor. kind: plain | affine | resid (centred action input)."""

    def __init__(self, kind="plain", ehat=None):
        super().__init__()
        self.kind = kind
        self.ehat = ehat  # callable s -> E[a|s] estimate (frozen)
        if kind == "affine":
            self.base = mlp(DS, DS)
            self.gain = mlp(DS, DS * DA)
        else:
            self.net = mlp(DS + DA, DS)

    def forward(self, s, a):
        if self.kind == "affine":
            return self.base(s) + (self.gain(s).view(*s.shape[:-1], DS, DA) @ a.unsqueeze(-1)).squeeze(-1)
        if self.kind == "resid":
            a = a - self.ehat(s)
        return self.net(torch.cat([s, a], -1))


class Orth(nn.Module):
    """Partialled-out effect model: D(s,a) = m(s) + h(s, a-ehat(s)) - mean_eps h(s, eps)."""

    def __init__(self, ehat, m, pool):
        super().__init__()
        self.ehat, self.m, self.pool = ehat, m, pool
        self.h = mlp(DS + DA, DS)

    def hres(self, s, e):
        return self.h(torch.cat([s, e], -1))

    def hbar(self, s, K=16, gen=None):
        idx = torch.randint(0, self.pool.shape[0], (K,), generator=gen)
        e = self.pool[idx]  # K,DA
        sK = s.unsqueeze(-2).expand(*s.shape[:-1], K, DS)
        eK = e.expand(*s.shape[:-1], K, DA)
        return self.hres(sK, eK).mean(-2)

    def forward(self, s, a):
        return self.m(s) + self.hres(s, a - self.ehat(s)) - self.hbar(s, gen=torch.Generator().manual_seed(0))


def fit_reg(net, X, Y, epochs=40, bs=128, lr=1e-3, wd=1e-4, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = F.mse_loss(net(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def make_ehat(S, A, seed):
    torch.manual_seed(seed)
    net = fit_reg(mlp(DS, DA, 64, 2), S, A, epochs=30, seed=seed)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def train(method, S, A, S2, seed, lam=1.0, epochs=40, bs=128, lr=1e-3, wd=1e-4, ehat=None):
    """Returns a callable model(s,a)->s' (differentiable-free, eval mode)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    D = S2 - S
    n = S.shape[0]
    if ehat is None:
        ehat = make_ehat(S, A, seed)
    res_sd = (A - ehat(S)).std(0)  # residual scale (for normalising IDM targets)

    if method in ("mle", "idm", "idm_delta", "nce"):
        P = Predictor("plain")
    elif method == "affine":
        P = Predictor("affine")
    elif method in ("cin", "ridm"):
        P = Predictor("resid", ehat)
    elif method == "orth":
        # cross-fitted m(s) = E[D|s]
        m_oof = torch.zeros_like(D)
        half = torch.arange(n) % 2 == 0
        mA = fit_reg(mlp(DS, DS), S[half], D[half], epochs=epochs, seed=seed)
        mB = fit_reg(mlp(DS, DS), S[~half], D[~half], epochs=epochs, seed=seed + 1)
        with torch.no_grad():
            m_oof[~half] = mA(S[~half]); m_oof[half] = mB(S[half])
        mfull = fit_reg(mlp(DS, DS), S, D, epochs=epochs, seed=seed + 2)
        for p in mfull.parameters():
            p.requires_grad_(False)
        pool = (A - ehat(S)).detach()
        P = Orth(ehat, mfull, pool)
    else:
        raise ValueError(method)

    extra = nn.ModuleDict()
    if method == "idm":
        extra["g"] = mlp(2 * DS, DA, 64, 2)
    if method == "idm_delta":
        extra["g"] = mlp(DS, DA, 64, 2)
    if method == "ridm":
        extra["g"] = mlp(2 * DS, DA, 64, 2)
    if method == "nce":
        extra["q"] = mlp(2 * DS, 32, 64, 2)
        extra["u"] = mlp(DA, 32, 64, 2)
    params = list(P.parameters()) + list(extra.parameters())
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            s, a, d = S[b], A[b], D[b]
            if method == "orth":
                e = a - ehat(s)
                tgt = d - m_oof[b]
                hb = P.hbar(s, K=8, gen=g)
                pred = P.hres(s, e) - hb
                loss = F.mse_loss(pred, tgt)
            else:
                dh = P(s, a)
                loss = F.mse_loss(dh, d)
                s2h = s + dh
                if method == "idm":
                    loss = loss + lam * (F.mse_loss(extra["g"](torch.cat([s, S2[b]], -1)), a)
                                         + F.mse_loss(extra["g"](torch.cat([s, s2h], -1)), a))
                elif method == "idm_delta":
                    loss = loss + lam * (F.mse_loss(extra["g"](d), a) + F.mse_loss(extra["g"](dh), a))
                elif method == "ridm":
                    e = (a - ehat(s)) / res_sd
                    loss = loss + lam * (F.mse_loss(extra["g"](torch.cat([s, S2[b]], -1)), e)
                                         + F.mse_loss(extra["g"](torch.cat([s, s2h], -1)), e))
                elif method == "nce":
                    z = extra["q"](torch.cat([s, s2h], -1))
                    u = extra["u"](a)
                    logits = z @ u.T / 0.5
                    lab = torch.arange(len(b))
                    loss = loss + lam * 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(); loss.backward(); opt.step()
    P.eval()

    def model(s, a):
        with torch.no_grad():
            return s + P(s, a)
    return model


# ---------------------------------------------------------------- evaluation
def action_metrics(env, model, S_eval, ehat, seed, n_pairs=2000):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, S_eval.shape[0], (n_pairs,), generator=g)
    s = S_eval[idx]
    out = {}
    for name in ("off", "on"):
        if name == "off":
            a1 = 2 * torch.rand(n_pairs, DA, generator=g) - 1
            a2 = 2 * torch.rand(n_pairs, DA, generator=g) - 1
        else:
            m = ehat(s)
            sd = (0.15)
            a1 = torch.clamp(m + sd * torch.randn(n_pairs, DA, generator=g), -1, 1)
            a2 = torch.clamp(m + sd * torch.randn(n_pairs, DA, generator=g), -1, 1)
        true = env.step(s, a1) - env.step(s, a2)
        est = model(s, a1) - model(s, a2)
        tn = true.norm(dim=-1)
        out[f"sens_{name}"] = float((est.norm(dim=-1) / tn.clamp_min(1e-8)).median())
        out[f"cerr_{name}"] = float((est - true).norm(dim=-1).mean() / tn.mean())
        cos = F.cosine_similarity(est, true, dim=-1)
        out[f"cos_{name}"] = float(cos.mean())
    return out


def cem_plan(step_fn, s0, goal, H=6, pop=128, elites=16, iters=4, seed=0, act_pen=0.01):
    P = s0.shape[0]
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros(P, H, DA)
    sd = 0.7 * torch.ones(P, H, DA)
    for _ in range(iters):
        acts = torch.clamp(mu.unsqueeze(1) + sd.unsqueeze(1) * torch.randn(P, pop, H, DA, generator=g), -1, 1)
        s = s0.unsqueeze(1).expand(P, pop, DS)
        for t in range(H):
            s = step_fn(s, acts[:, :, t])
        cost = ((s - goal.unsqueeze(1)) ** 2).sum(-1) + act_pen * (acts ** 2).sum((-1, -2))
        top = cost.topk(elites, dim=1, largest=False).indices
        el = torch.gather(acts, 1, top.view(P, elites, 1, 1).expand(P, elites, H, DA))
        mu, sd = el.mean(1), el.std(1) + 0.05
    return mu  # P,H,DA


def true_cost(env, s0, goal, plan, act_pen=0.01):
    s = s0
    for t in range(plan.shape[1]):
        s = env.step(s, plan[:, t])
    return ((s - goal) ** 2).sum(-1) + act_pen * (plan ** 2).sum((-1, -2))


def make_problems(env, S_pool, n_prob, H, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, S_pool.shape[0], (n_prob,), generator=g)
    s0 = S_pool[idx]
    a = 2 * torch.rand(n_prob, H, DA, generator=g) - 1
    s = s0
    for t in range(H):
        s = env.step(s, a[:, t])
    return s0, s  # goal reachable by a random plan


def planning_eval(env, model, s0, goal, H=6, seed=0, oracle_plan=None):
    plan = cem_plan(model, s0, goal, H=H, seed=seed)
    c = true_cost(env, s0, goal, plan)
    return c, plan
