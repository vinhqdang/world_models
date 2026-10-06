"""Small dynamical-systems testbed for long-horizon rollout drift.

Systems are simulated exactly (RK4 at a fine step), observed through a fixed
smooth random embedding into R^16, and standardised.  A learned predictor acts
on the embedded observation (the 'latent').  Ground truth is therefore known for
every rollout metric.
"""
import numpy as np, torch, torch.nn as nn

D_OBS = 16


def _rk4(f, x, dt, sub):
    h = dt / sub
    for _ in range(sub):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def lorenz(x):
    s, r, b = 10.0, 28.0, 8.0 / 3
    return np.stack([s * (x[..., 1] - x[..., 0]), x[..., 0] * (r - x[..., 2]) - x[..., 1],
                     x[..., 0] * x[..., 1] - b * x[..., 2]], -1)


def vdp(x, mu=2.0):
    return np.stack([x[..., 1], mu * (1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], -1)


SYS = {
    "lorenz": dict(f=lorenz, dim=3, dt=0.02, sub=4, spin=1000, lyap=0.906, T=400, thr_T=55),
    "vdp": dict(f=vdp, dim=2, dt=0.2, sub=20, spin=0, lyap=0.0, T=300, thr_T=38),
}


def embedding(dim, seed=7):
    g = np.random.RandomState(seed)
    W1 = g.randn(dim, 32) / np.sqrt(dim) * 0.5; b1 = g.randn(32) * 0.3
    W2 = g.randn(32, D_OBS) / np.sqrt(32) * 0.9
    return lambda s: np.tanh(np.tanh(s @ W1 + b1) @ W2)


def sample_ic(name, n, rng, wide=False):
    if name == "lorenz":
        return rng.randn(n, 3) * np.array([8, 8, 8]) + np.array([0, 0, 25])
    if wide:  # basin-wide start for vdp (training data includes transients)
        return rng.uniform(-3, 3, size=(n, 2))
    return rng.uniform(-3, 3, size=(n, 2))


def simulate(name, n_traj, T, seed, on_attractor=True, wide=False):
    """returns states (n,T+1,dim). For lorenz and for vdp-test the spin-up is discarded."""
    S = SYS[name]; rng = np.random.RandomState(seed)
    x = sample_ic(name, n_traj, rng, wide)
    spin = S["spin"] if on_attractor else 0
    for _ in range(spin):
        x = _rk4(S["f"], x, S["dt"], S["sub"])
    out = [x]
    for _ in range(T):
        x = _rk4(S["f"], x, S["dt"], S["sub"]); out.append(x)
    return np.stack(out, 1)


class Data:
    """train / val / test trajectories in standardised embedded coordinates."""

    def __init__(self, name, seed, n_train=60, T_train=300, n_test=100, role="test"):
        S = SYS[name]; self.name = name; self.S = S
        emb = embedding(S["dim"])
        # train data: lorenz on attractor; vdp = transients + cycle (spin 0)
        tr = simulate(name, n_train, T_train, seed * 7 + 1, on_attractor=(name == "lorenz"), wide=True)
        self.emb = emb
        flat = emb(tr.reshape(-1, S["dim"]))
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        nrm = lambda s: ((emb(s) - self.mu) / self.sd).astype(np.float32)
        self.nrm = nrm
        self.train_states = tr
        self.train = torch.tensor(nrm(tr))                    # (n,T+1,16)
        # test: on-attractor initial conditions (vdp: spin 20 time units to reach the cycle)
        base = 10_000 if role == "test" else 20_000
        if name == "lorenz":
            te = simulate(name, n_test, S["T"], base + seed, True)
        else:
            rng = np.random.RandomState(base + seed)
            x = rng.uniform(-3, 3, size=(n_test, 2))
            for _ in range(200): x = _rk4(S["f"], x, S["dt"], S["sub"])
            out = [x]
            for _ in range(S["T"]): x = _rk4(S["f"], x, S["dt"], S["sub"]); out.append(x)
            te = np.stack(out, 1)
        self.test = torch.tensor(nrm(te)); self.test_states = te
        # long true reference for invariant-measure fidelity
        ref = simulate(name, 8, 1500, base + 500 + seed, True) if name == "lorenz" else te[:, ::1][:, :].reshape(n_test, -1, S["dim"])
        if name == "vdp":
            rng = np.random.RandomState(base + 900 + seed); x = rng.uniform(-3, 3, size=(8, 2))
            for _ in range(200): x = _rk4(S["f"], x, S["dt"], S["sub"])
            out = [x]
            for _ in range(1500): x = _rk4(S["f"], x, S["dt"], S["sub"]); out.append(x)
            ref = np.stack(out, 1)
        self.ref = torch.tensor(nrm(ref.reshape(-1, S["dim"]))).float()


class ResMLP(nn.Module):
    def __init__(self, d=D_OBS, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)


class AE(nn.Module):
    def __init__(self, d=D_OBS, k=4, h=64):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, k))
        self.d = nn.Sequential(nn.Linear(k, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, z):
        return self.d(self.e(z))


def train_ae(data, steps=3000, k=4, noise=0.05, seed=0):
    torch.manual_seed(seed)
    ae = AE(k=k); opt = torch.optim.Adam(ae.parameters(), 2e-3)
    X = data.train.reshape(-1, D_OBS)
    for i in range(steps):
        z = X[torch.randint(0, len(X), (512,))]
        loss = ((ae(z + noise * torch.randn_like(z)) - z) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return ae


@torch.no_grad()
def rollout(model, z0, T, proj=None, clip=1e3):
    z = z0; out = [z]
    for _ in range(T):
        z = model(z)
        if proj is not None: z = proj(z)
        z = torch.nan_to_num(z, nan=clip, posinf=clip, neginf=-clip).clamp(-clip, clip)
        out.append(z)
    return torch.stack(out, 1)


def energy_dist(X, Y, n=1500, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = X[torch.randperm(len(X), generator=g)[:n]]; Y = Y[torch.randperm(len(Y), generator=g)[:n]]
    xy = torch.cdist(X, Y).mean(); xx = torch.cdist(X, X).sum() / (n * (n - 1)); yy = torch.cdist(Y, Y).sum() / (n * (n - 1))
    return float(2 * xy - xx - yy)


def evaluate(model, data, proj=None, thr=0.3, long_T=3000, n_long=16, seed=0):
    S = data.S
    te = data.test
    with torch.no_grad():
        pr = rollout(model, te[:, 0], te.shape[1] - 1, proj)
        err = ((pr - te) ** 2).mean(-1).sqrt()                      # (n,T+1) RMS per-dim, standardised
        bad = err > thr
        first = torch.where(bad.any(1), bad.float().argmax(1), torch.full((len(te),), te.shape[1]))
        vpt = first.float()
        # long rollouts for statistics
        z0 = te[:n_long, 0]
        lr = rollout(model, z0, long_T, proj)
        blown = (lr.abs().amax((1, 2)) > 8)                         # left the data range (embedded data is ~N(0,1) per dim)
        keep = lr[~blown][:, 200:].reshape(-1, D_OBS) if (~blown).any() else None
    ed = energy_dist(keep, data.ref, seed=seed) if keep is not None and len(keep) > 2000 else float("nan")
    # off-manifold distance: nearest true-reference sample
    if keep is not None and len(keep) > 0:
        sub = keep[torch.randperm(len(keep))[:2000]]
        offm = float(torch.cdist(sub, data.ref).min(1).values.mean())
    else:
        offm = float("nan")
    return dict(vpt=float(vpt.mean()), vpt_med=float(vpt.median()), vpt_lyap=float(vpt.mean()) * S["dt"] * S["lyap"],
                blow=float(blown.float().mean()), edist=ed, offman=offm)
