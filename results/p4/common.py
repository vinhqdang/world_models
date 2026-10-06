import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

# ---- tiny 2-D point world with position-dependent gain (breaks translation symmetry) ----
def step(s, a):
    g = np.stack([1.0 + 0.5 * np.sin(3 * s[..., 1]), 1.0 + 0.5 * np.cos(3 * s[..., 0])], -1)
    return np.clip(s + 0.12 * a * g, -1, 1)

def step_t(s, a):
    g = torch.stack([1.0 + 0.5 * torch.sin(3 * s[..., 1]), 1.0 + 0.5 * torch.cos(3 * s[..., 0])], -1)
    return torch.clamp(s + 0.12 * a * g, -1, 1)

class ObsMap:
    """fixed random injective-ish observation map  s(2) -> obs(D)"""
    def __init__(self, D, seed, kind):
        r = np.random.RandomState(seed)
        self.A = r.randn(D, 2) * (1.2 if kind == 'tanh' else 1.6)
        self.b = r.randn(D) * 0.3
        self.kind = kind
    def __call__(self, s):
        z = s @ self.A.T + self.b
        return np.tanh(z) if self.kind == 'tanh' else np.sin(z)

def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))

def collect(n_traj, K, seed):
    r = np.random.RandomState(seed)
    s = r.uniform(-1, 1, (n_traj, 2)); S = [s]; A = []
    for _ in range(K):
        a = r.uniform(-1, 1, (n_traj, 2)); s = step(s, a); S.append(s); A.append(a)
    return np.stack(S, 1), np.stack(A, 1)   # (n,K+1,2), (n,K,2)

def vic_reg(z, var_w=1.0, cov_w=0.04):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    v = torch.relu(1.0 - std).pow(2).mean()
    c = (z.T @ z) / (z.shape[0] - 1)
    off = c - torch.diag(torch.diag(c))
    return var_w * v + cov_w * off.pow(2).sum() / z.shape[1]

def sliced_w2(x, y, n_proj=64):
    d = x.shape[1]
    th = torch.randn(d, n_proj); th = th / th.norm(dim=0, keepdim=True)
    px = (x @ th).sort(0)[0]; py = (y @ th).sort(0)[0]
    m = min(px.shape[0], py.shape[0])
    ix = torch.linspace(0, px.shape[0] - 1, m).long(); iy = torch.linspace(0, py.shape[0] - 1, m).long()
    return (px[ix] - py[iy]).pow(2).mean()

def train_jepa(obs_fn, d, n_traj, K, seed, steps=2500, lr=2e-3, reg_w=1.0):
    """fresh JEPA: encoder E(o), predictor P(z,a); VICReg-lite anti-collapse; K-step rollout loss; stop-grad-free with reg."""
    torch.manual_seed(seed)
    S, A = collect(n_traj, K, seed + 1000)
    O = torch.tensor(obs_fn(S.reshape(-1, 2)).reshape(n_traj, K + 1, -1), dtype=torch.float32); A = torch.tensor(A, dtype=torch.float32)
    E = mlp(O.shape[-1], d); P = mlp(d + 2, d)
    opt = torch.optim.Adam(list(E.parameters()) + list(P.parameters()), lr=lr)
    for it in range(steps):
        idx = torch.randint(0, n_traj, (256,))
        o, a = O[idx], A[idx]
        z = E(o.reshape(-1, o.shape[-1])).reshape(256, K + 1, d)
        loss = 0; zh = z[:, 0]
        for k in range(K):
            zh = zh + P(torch.cat([zh, a[:, k]], -1))
            loss = loss + (zh - z[:, k + 1]).pow(2).mean()
        loss = loss / K + reg_w * vic_reg(z.reshape(-1, d))
        opt.zero_grad(); loss.backward(); opt.step()
    return E, P

class Pred(nn.Module):   # residual predictor wrapper used everywhere
    def __init__(self, P): super().__init__(); self.P = P
    def forward(self, z, a): return z + self.P(torch.cat([z, a], -1))

@torch.no_grad()
def cem_plan(P, z0, zg, H=8, N=200, iters=4, elite=20, rng=None):
    mu = torch.zeros(H, 2); sd = torch.ones(H, 2) * 0.7
    for _ in range(iters):
        a = (mu + sd * torch.randn(N, H, 2, generator=rng)).clamp(-1, 1)
        z = z0.expand(N, -1)
        for h in range(H):
            z = z + P(torch.cat([z, a[:, h]], -1))
        c = (z - zg).pow(2).sum(-1)
        ei = c.topk(elite, largest=False).indices
        mu = a[ei].mean(0); sd = a[ei].std(0) + 0.05
    return mu[0]

@torch.no_grad()
def run_episodes(E, P, obs_fn, n_ep, seed, T=25, tol=0.15, dmin=0.5, dmax=1.2):
    """closed-loop MPC in the true env using encoder E on observations; success if true state within tol of goal."""
    r = np.random.RandomState(seed); rng = torch.Generator().manual_seed(seed); succ = 0; fin = []
    for _ in range(n_ep):
        while True:
            s = r.uniform(-1, 1, 2); g = r.uniform(-1, 1, 2)
            if dmin <= np.linalg.norm(s - g) <= dmax: break
        zg = E(torch.tensor(obs_fn(g[None]), dtype=torch.float32))
        for t in range(T):
            z0 = E(torch.tensor(obs_fn(s[None]), dtype=torch.float32))
            a = cem_plan(P, z0, zg, rng=rng).numpy()
            s = step(s, a)
        fin.append(np.linalg.norm(s - g)); succ += fin[-1] < tol
    return succ / n_ep, float(np.mean(fin))
