"""State-level mean predictor mu(H, a) and its 2-fold cross-fitting.

H_t = (s_{t-2}, s_{t-1}, s_t) (first state repeated when unavailable). The network predicts the standardized change
s_{t+1} - s_t of all recorded coordinates from standardized H and the action; trained by AdamW on the factual
transitions of all steps of the episodes, early-stopped on held-out factual episodes (no counterfactual quantity enters).
"""
import copy

import numpy as np
import torch
import torch.nn as nn


def contexts(states, t_idx=None):
    """states (n, T+1, D) -> H (n, T, 3 D) for transitions t = 0..T-1 (or the given t's)."""
    n, T1, D = states.shape
    T = T1 - 1
    ts = list(range(T)) if t_idx is None else list(t_idx)
    H = np.stack([np.concatenate([states[:, max(t - 2, 0)], states[:, max(t - 1, 0)], states[:, t]], -1) for t in ts], 1)
    return H


def flatten_transitions(col, t_max=None):
    """Collection dict -> H (M, 3D), a (M, p), y (M, D) next state, s (M, D) current, episode id (M,), step (M,)."""
    S, A = col["states"], col["actions"]
    n, T = A.shape[:2]
    T = T if t_max is None else min(T, t_max)
    H = contexts(S)[:, :T]
    ep = np.repeat(np.arange(n)[:, None], T, 1)
    st = np.repeat(np.arange(T)[None], n, 0)
    return (H.reshape(n * T, -1), A[:, :T].reshape(n * T, -1), S[:, 1:T + 1].reshape(n * T, -1),
            S[:, :T].reshape(n * T, -1), ep.reshape(-1), st.reshape(-1))


class Scaler:
    def __init__(self, H, a, y, s):
        self.hm, self.hs = H.mean(0), H.std(0) + 1e-6
        self.am, self.as_ = a.mean(0), a.std(0) + 1e-6
        d = y - s
        self.dm, self.ds = d.mean(0), d.std(0) + 1e-8

    def inp(self, H, a):
        return np.concatenate([(H - self.hm) / self.hs, (a - self.am) / self.as_], -1)


def mlp(i, o, width=512, depth=3, act=nn.SiLU):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class MeanModel:
    """mu(H, a): physical next state = current state + delta."""

    def __init__(self, scaler, net, D):
        self.sc, self.net, self.D = scaler, net, D

    @torch.no_grad()
    def predict(self, H, a, s):
        x = torch.as_tensor(self.sc.inp(H, a), dtype=torch.float32)
        out = []
        for i in range(0, len(x), 8192):
            out.append(self.net(x[i:i + 8192]).numpy())
        d = np.concatenate(out) * self.sc.ds + self.sc.dm
        return s + d

    def torch_fn(self):
        """Differentiable map (H, a, s) torch tensors in physical units -> next state (for abduction baselines)."""
        sc, net = self.sc, self.net
        hm, hs = torch.as_tensor(sc.hm, dtype=torch.float32), torch.as_tensor(sc.hs, dtype=torch.float32)
        am, as_ = torch.as_tensor(sc.am, dtype=torch.float32), torch.as_tensor(sc.as_, dtype=torch.float32)
        dm, ds = torch.as_tensor(sc.dm, dtype=torch.float32), torch.as_tensor(sc.ds, dtype=torch.float32)

        def f(H, a, s):
            return s + net(torch.cat([(H - hm) / hs, (a - am) / as_], -1)) * ds + dm
        return f


def fit_mean(H, a, y, s, ep, seed, epochs=40, width=512, depth=3, lr=1e-3, wd=1e-4, batch=256, val_frac=0.1,
             threads=2, verbose=False):
    """Fit one predictor; the last ``val_frac`` of episodes (random) is the early-stopping validation set."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    eps_ids = np.unique(ep)
    val_eps = set(rng.choice(eps_ids, max(1, int(val_frac * len(eps_ids))), replace=False).tolist())
    is_val = np.array([e in val_eps for e in ep])
    sc = Scaler(H[~is_val], a[~is_val], y[~is_val], s[~is_val])
    X = torch.as_tensor(sc.inp(H, a), dtype=torch.float32)
    Y = torch.as_tensor(((y - s) - sc.dm) / sc.ds, dtype=torch.float32)
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    torch.manual_seed(seed)
    net = mlp(X.shape[1], Y.shape[1], width, depth)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state = np.inf, None
    for e in range(epochs):
        perm = rng.permutation(tr)
        net.train()
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(X[va]) - Y[va]) ** 2).mean().item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(net.state_dict())
        if verbose:
            print(e, vl, flush=True)
    net.load_state_dict(best_state)
    net.eval()
    return MeanModel(sc, net, y.shape[1]), best


def crossfit(H, a, y, s, ep, seed, **kw):
    """2-fold (by episode) cross-fitted predictions mu_oof for every transition, plus a model on all data."""
    rng = np.random.default_rng(10_000 + seed)
    eps_ids = np.unique(ep)
    fold_of = dict(zip(eps_ids.tolist(), rng.integers(0, 2, len(eps_ids)).tolist()))
    fold = np.array([fold_of[e] for e in ep])
    oof = np.empty_like(y)
    models = []
    for k in (0, 1):
        m, _ = fit_mean(H[fold != k], a[fold != k], y[fold != k], s[fold != k], ep[fold != k], seed * 7 + k, **kw)
        oof[fold == k] = m.predict(H[fold == k], a[fold == k], s[fold == k])
        models.append(m)
    full, vl = fit_mean(H, a, y, s, ep, seed * 7 + 2, **kw)
    return dict(oof=oof, fold=fold, full=full, folds=models, val_loss=vl)
