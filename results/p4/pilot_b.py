"""Pilot B (kill test): which transitions to collect so that model-based planning succeeds?
2-D point robot, central wall with detours (a discontinuity in the dynamics: blocked moves leave the state unchanged).
Acquisition strategies, equal budget per round:
  random      : uniform (s,a)
  disagree    : top ensemble-variance (s,a) from a random pool (one-step prediction disagreement)
  dagger      : run the current planner (ensemble mean) on random tasks with epsilon-random actions (planner-distribution data)
  crossregret : decision-focused. For a pool of (s,g) tasks, each ensemble member plans; score = max_{i,j} [J_j(a_i)-J_j(a_j)]
                (regret member j would suffer executing member i's plan); execute the plan of the most optimistic member from the
                top-scoring tasks (real transitions), so data lands where models disagree about WHICH plan is best.
"""
import sys, json, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
H = 14; T = 30; TOL = 0.12; STEP = 0.12
def env_step(s, a):
    """s,a: (...,2) numpy. wall: x=0, |y|<0.4 ; crossing it is blocked (jump discontinuity)."""
    sn = np.clip(s + STEP * a, -1, 1)
    cross = (s[..., 0] * sn[..., 0] < 0) | (np.abs(sn[..., 0]) < 1e-9)
    t = np.where(np.abs(sn[..., 0] - s[..., 0]) > 1e-12, -s[..., 0] / (sn[..., 0] - s[..., 0] + 1e-12), 0.0)
    yc = s[..., 1] + t * (sn[..., 1] - s[..., 1])
    blocked = cross & (np.abs(yc) < 0.4)
    return np.where(blocked[..., None], s, sn)
def sample_task(r):
    while True:
        s = np.array([r.uniform(-0.55, -0.05), r.uniform(-0.3, 0.3)]); g = np.array([r.uniform(0.05, 0.55), r.uniform(-0.3, 0.3)])
        return s, g
class Ens(nn.Module):
    def __init__(s, K=4, h=128):
        super().__init__(); s.K = K
        s.nets = nn.ModuleList([nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2)) for _ in range(K)])
    def member(s, k, x, a): return x + 0.12 * s.nets[k](torch.cat([x, a], -1))   # delta param. scaled
    def mean(s, x, a): return torch.stack([s.member(k, x, a) for k in range(s.K)]).mean(0)
def fit(ens, D, seed, epochs=150):
    S, A, S2 = [torch.tensor(np.array(v), dtype=torch.float32) for v in D]
    n = len(S); g = torch.Generator().manual_seed(seed)
    for k, net in enumerate(ens.nets):
        for p in net.parameters(): nn.init.normal_(p, 0, 0.1) if p.dim() > 1 and False else None
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        idx_b = torch.randint(0, n, (n,), generator=torch.Generator().manual_seed(seed * 100 + k))   # bootstrap
        Sb, Ab, S2b = S[idx_b], A[idx_b], S2[idx_b]
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 256):
                j = perm[i:i + 256]
                loss = (Sb[j] + 0.12 * net(torch.cat([Sb[j], Ab[j]], -1)) - S2b[j]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def cem_batch(f, s0, g, N=100, iters=3, elite=10, rng=None):
    """f(x,a)->x'; s0,g: (B,2). returns plans (B,H,2) and their model costs (B,)"""
    B = s0.shape[0]; mu = torch.zeros(B, H, 2); sd = torch.ones(B, H, 2) * 0.7
    for _ in range(iters):
        a = (mu[:, None] + sd[:, None] * torch.randn(B, N, H, 2, generator=rng)).clamp(-1, 1)
        x = s0[:, None].expand(B, N, 2); c = 0
        for h in range(H):
            x = f(x, a[:, :, h]); c = c + 0.1 * (x - g[:, None]).pow(2).sum(-1)
        c = c + 1.0 * (x - g[:, None]).pow(2).sum(-1)
        ei = c.topk(elite, dim=1, largest=False).indices
        ae = torch.gather(a, 1, ei[:, :, None, None].expand(B, elite, H, 2))
        mu = ae.mean(1); sd = ae.std(1) + 0.05
    return mu, None
@torch.no_grad()
def plan_cost(f, s0, g, plan):
    x = s0; c = 0
    for h in range(H):
        x = f(x, plan[:, h]); c = c + 0.1 * (x - g).pow(2).sum(-1)
    return c + 1.0 * (x - g).pow(2).sum(-1)
@torch.no_grad()
def evaluate(ens, tasks, seed):
    rng = torch.Generator().manual_seed(seed); succ = 0
    S = np.array([t[0] for t in tasks]); G = np.array([t[1] for t in tasks])
    s = S.copy(); g = torch.tensor(G, dtype=torch.float32)
    for t in range(T):
        plan, _ = cem_batch(ens.mean, torch.tensor(s, dtype=torch.float32), g, rng=rng)
        s = env_step(s, plan[:, 0].numpy())
    return float((np.linalg.norm(s - G, axis=1) < TOL).mean())
def rollout_real(s, plan):
    out = []
    for h in range(plan.shape[0]):
        s2 = env_step(s, plan[h]); out.append((s.copy(), plan[h].copy(), s2.copy())); s = s2
    return out
def run(strategy, seed, rounds=5, B=204, n0=300):
    r = np.random.RandomState(seed); torch.manual_seed(seed); rng = torch.Generator().manual_seed(seed + 5)
    # initial random data from the whole arena (uniform)
    S0 = r.uniform(-1, 1, (n0, 2)); A0 = r.uniform(-1, 1, (n0, 2)); D = [list(S0), list(A0), list(env_step(S0, A0))]
    tasks = [sample_task(np.random.RandomState(9999 + i)) for i in range(80)]
    ens = Ens(); fit(ens, D, seed); curve = [(len(D[0]), evaluate(ens, tasks, seed))]
    for rd in range(rounds):
        if strategy == 'random':
            S = r.uniform(-1, 1, (B, 2)); A = r.uniform(-1, 1, (B, 2)); new = list(zip(S, A, env_step(S, A)))
        elif strategy == 'random_local':   # random but restricted to the task region (oracle knowledge of task support)
            S = np.stack([r.uniform(-0.6, 0.6, B), r.uniform(-0.45, 0.45, B)], 1); A = r.uniform(-1, 1, (B, 2)); new = list(zip(S, A, env_step(S, A)))
        elif strategy == 'disagree':
            S = np.stack([r.uniform(-0.6, 0.6, 3000), r.uniform(-0.45, 0.45, 3000)], 1); A = r.uniform(-1, 1, (3000, 2))
            with torch.no_grad():
                St, At = torch.tensor(S, dtype=torch.float32), torch.tensor(A, dtype=torch.float32)
                P = torch.stack([ens.member(k, St, At) for k in range(ens.K)]); v = P.var(0).sum(-1)
            top = v.topk(B).indices.numpy(); new = list(zip(S[top], A[top], env_step(S[top], A[top])))
        elif strategy == 'dagger':
            nt = B // H; new = []
            for _ in range(nt):
                s, g = sample_task(r)
                for h in range(H):
                    plan, _ = cem_batch(ens.mean, torch.tensor(s[None], dtype=torch.float32), torch.tensor(g[None], dtype=torch.float32), rng=rng)
                    a = plan[0, 0].numpy()
                    if r.rand() < 0.3: a = r.uniform(-1, 1, 2)
                    s2 = env_step(s, a); new.append((s.copy(), a.copy(), s2.copy())); s = s2
        elif strategy == 'crossregret':
            M = 150; tk = [sample_task(r) for _ in range(M)]
            s0 = torch.tensor(np.array([t[0] for t in tk]), dtype=torch.float32); g = torch.tensor(np.array([t[1] for t in tk]), dtype=torch.float32)
            plans = [cem_batch(lambda x, a, k=k: ens.member(k, x, a), s0, g, rng=rng)[0] for k in range(ens.K)]
            J = np.zeros((ens.K, ens.K, M))      # J[j,i] = cost under member j of plan i
            for j in range(ens.K):
                for i in range(ens.K):
                    J[j, i] = plan_cost(lambda x, a, j=j: ens.member(j, x, a), s0, g, plans[i]).numpy()
            reg = np.zeros(M)
            for i in range(ens.K):
                for j in range(ens.K): reg = np.maximum(reg, J[j, i] - J[j, j])
            opt_member = np.argmin(np.array([J[i, i] for i in range(ens.K)]), axis=0)   # most optimistic member per task
            top = np.argsort(-reg)[:B // H]; new = []
            for m in top:
                new += rollout_real(tk[m][0], plans[opt_member[m]][m].numpy())
        for k in range(3): D[k] += [x[k] for x in new]
        ens = Ens(); fit(ens, D, seed + rd + 1); curve.append((len(D[0]), evaluate(ens, tasks, seed)))
    return curve
if __name__ == '__main__':
    seed = int(sys.argv[1]); strategies = sys.argv[2].split(','); out = {}
    t0 = time.time()
    for st in strategies:
        out[st] = run(st, seed); print(seed, st, [round(c[1], 3) for c in out[st]], round(time.time() - t0), flush=True)
    json.dump(out, open(f'pilot_b_s{seed}_{"_".join(strategies)}.json', 'w'))
