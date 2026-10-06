"""Training rules for latent one-step predictors (all share the ResMLP architecture)."""
import torch, numpy as np
from selwm.p2_dyn import ResMLP, D_OBS


def make_pairs(data):
    tr = data.train
    return tr[:, :-1].reshape(-1, D_OBS), tr[:, 1:].reshape(-1, D_OBS)


@torch.no_grad()
def local_tangent_normal(X, d_tan, k=32, chunk=1024):
    """Local-PCA tangent basis for each row of X; returns normal projector factors (N, D, d_tan) tangents."""
    N = len(X); T = torch.empty(N, D_OBS, d_tan)
    for i in range(0, N, chunk):
        xb = X[i:i + chunk]
        idx = torch.cdist(xb, X).topk(k, largest=False).indices
        nb = X[idx]; nb = nb - nb.mean(1, keepdim=True)
        C = nb.transpose(1, 2) @ nb
        ev, V = torch.linalg.eigh(C)
        T[i:i + chunk] = V[:, :, -d_tan:]
    return T


def normal_noise(T, sigma):
    n = torch.randn(T.shape[0], D_OBS)
    tang = (T @ (T.transpose(1, 2) @ n.unsqueeze(-1))).squeeze(-1)
    return sigma * (n - tang)


def fit(data, mode, seed=0, steps=4000, bs=512, lr=1e-3, sigma=0.0, H=8, ss_p=0.5, tangents=None, model=None,
        gamma=1.0, verbose=False):
    """mode in: tf, iso, normal, ss (multi-step BPTT with scheduled sampling), lml (tangent-linear multi-step loss)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model or ResMLP()
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr = data.train; n, Tp1, _ = tr.shape
    X, Y = make_pairs(data)
    for it in range(steps):
        if mode in ("tf", "iso", "normal"):
            idx = torch.randint(0, len(X), (bs,))
            x, y = X[idx], Y[idx]
            if mode == "iso": x = x + sigma * torch.randn_like(x)
            elif mode == "normal": x = x + normal_noise(tangents[idx], sigma)
            loss = ((model(x) - y) ** 2).mean()
        elif mode == "ss":
            tr_i = torch.randint(0, n, (bs // 4,)); t0 = torch.randint(0, Tp1 - H, (bs // 4,))
            seg = torch.stack([tr[a, b:b + H + 1] for a, b in zip(tr_i.tolist(), t0.tolist())])
            p = ss_p * min(1.0, it / (0.3 * steps))             # scheduled sampling ramp
            z = seg[:, 0]; loss = 0
            for h in range(H):
                zn = model(z); loss = loss + ((zn - seg[:, h + 1]) ** 2).mean()
                use_true = (torch.rand(len(z), 1) > p).float()
                z = use_true * seg[:, h + 1] + (1 - use_true) * zn
            loss = loss / H
        elif mode == "lml":
            tr_i = torch.randint(0, n, (bs // 4,)); t0 = torch.randint(0, Tp1 - H, (bs // 4,))
            seg = torch.stack([tr[a, b:b + H + 1] for a, b in zip(tr_i.tolist(), t0.tolist())])   # (B,H+1,D)
            B = seg.shape[0]
            zin = seg[:, :H].reshape(-1, D_OBS)
            delta = (model(zin) - seg[:, 1:].reshape(-1, D_OBS)).reshape(B, H, D_OBS)
            zj = zin.detach()
            J = torch.func.vmap(torch.func.jacrev(lambda v: model(v)))(zj).detach().reshape(B, H, D_OBS, D_OBS)
            e = delta[:, 0]; loss = (e ** 2).mean()
            for h in range(1, H):
                e = (J[:, h] @ e.unsqueeze(-1)).squeeze(-1) + delta[:, h]
                loss = loss + gamma ** h * (e ** 2).mean()
            loss = loss / H
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if verbose and it % 500 == 0: print(it, float(loss))
    return model


@torch.no_grad()
def rollout_deviation(model, data, tangents_X, Hc=20, n=2000):
    """RMS normal-component deviation of free-running states from nearest training point (self-calibration signal)."""
    tr = data.train; idx = torch.randint(0, tr.shape[0], (n,)); t0 = torch.randint(0, tr.shape[1] - 1, (n,))
    z = tr[idx, t0]; X = data.train.reshape(-1, D_OBS)
    devs = []
    for h in range(Hc):
        z = model(z)
        nn_d = torch.cdist(z, X)
        m = nn_d.min(1)
        devs.append(m.values)
    return torch.cat(devs).pow(2).mean().sqrt() / (D_OBS ** 0.5)     # per-dim RMS distance to the data set


@torch.no_grad()
def normal_deviation(model, data, T, Hc=20, n=3000):
    """RMS (per-dim) normal-space distance of free-running states to their nearest training state (self-calibration signal)."""
    tr = data.train; X = tr.reshape(-1, D_OBS)
    idx = torch.randint(0, tr.shape[0], (n,)); t0 = torch.randint(0, tr.shape[1] - 1, (n,))
    z = tr[idx, t0]; acc = []
    for h in range(Hc):
        z = model(z).clamp(-20, 20)
        j = torch.cdist(z, X).argmin(1)
        v = z - X[j]; Tj = T[j]
        vn = v - (Tj @ (Tj.transpose(1, 2) @ v.unsqueeze(-1))).squeeze(-1)
        acc.append(vn.pow(2).mean(1))
    return float(torch.cat(acc).mean().sqrt())


def fit_scn(data, T, seed=0, steps=4000, rounds=3, sigma0=0.02, cap=0.3, lr=2e-3, bs=1024):
    """Self-calibrated normal-space noise: noise scale is set to the model's own free-running normal deviation (fixed point)."""
    sig = sigma0; hist = [sig]
    m = fit(data, "normal", seed=seed, steps=steps, lr=lr, bs=bs, sigma=sig, tangents=T)
    for r in range(rounds):
        dev = normal_deviation(m, data, T)
        sig = min(cap, max(sigma0, dev)); hist.append(sig)
        m = fit(data, "normal", seed=seed + 1000 * (r + 1), steps=steps // 2, lr=lr / 2, bs=bs, sigma=sig, tangents=T, model=m)
    return m, hist
