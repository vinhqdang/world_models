"""Pilot A2 (mechanism test of the identifiability claim): on a TRANSLATION-INVARIANT latent world (torus, constant gain) the
dynamics-consistency equation has a 2-parameter family of solutions (psi(s)=phi(s+c)); on the broken-symmetry world (pilot A) it has none.
Prediction: anchor-free adaptation recovers phi only up to translation c (raw recovery bad, translation-aligned recovery good);
2 paired anchors (2 = dim of the symmetry group) or even 1 fix c."""
import sys, json, time, numpy as np, torch
import common
from common import *
seed = int(sys.argv[1]); d = 4; K = 5
def tstep(s, a): return (s + 0.12 * a + 1.0) % 2.0 - 1.0
common.step = tstep
class TorusObs:
    def __init__(self, D, seed, kind):
        r = np.random.RandomState(seed); self.A = r.randn(D, 4) * 0.9; self.b = r.randn(D) * 0.3; self.kind = kind
    def __call__(self, s):
        f = np.concatenate([np.cos(np.pi * s), np.sin(np.pi * s)], -1); z = f @ self.A.T + self.b
        return np.tanh(z) if self.kind == 'tanh' else np.sin(z)
src = TorusObs(8, 1, 'tanh'); tgt = TorusObs(10, 7, 'sin')
t0 = time.time()
E_src, P = train_jepa(src, d, 2000, K, seed, steps=2500)
for p in list(P.parameters()) + list(E_src.parameters()): p.requires_grad_(False)
Pf = lambda z, a: z + P(torch.cat([z, a], -1))
Sx = np.random.RandomState(seed + 77).uniform(-1, 1, (800, 2))
z_src_fn = lambda S: E_src(torch.tensor(src(S), dtype=torch.float32))
z0 = z_src_fn(Sx); V = z0.var(0).sum()
cs = np.array([[cx, cy] for cx in np.linspace(-1, 1, 21)[:-1] for cy in np.linspace(-1, 1, 21)[:-1]])
shift_tab = [((Sx + c + 1) % 2 - 1) for c in cs]
def recov(E):
    with torch.no_grad():
        z1 = E(torch.tensor(tgt(Sx), dtype=torch.float32))
        raw = float(((z1 - z0).pow(2).sum(-1).mean() / V))
        al = min(float(((z1 - z_src_fn(sh)).pow(2).sum(-1).mean() / V)) for sh in shift_tab)
        return raw, al
# source-side sanity: is E_src itself approx. consistent with P (one-step & 5-step) on fresh data?
S, A = collect(500, K, 31337); zz = z_src_fn(S.reshape(-1, 2)).reshape(500, K + 1, d); zh = zz[:, 0]; 
with torch.no_grad():
    for k in range(K): zh = Pf(zh, torch.tensor(A[:, k], dtype=torch.float32))
    src_cons = float((zh - zz[:, K]).pow(2).sum(-1).mean() / V)
Spaired = np.random.RandomState(seed + 9).uniform(-1, 1, (500, 2)); Zpair = z_src_fn(Spaired); Opair = torch.tensor(tgt(Spaired), dtype=torch.float32)
def adapt(npair, ntr=1000, steps=2500):
    torch.manual_seed(seed + 3)
    S, A = collect(ntr // K, K, seed + 2000)
    O = torch.tensor(tgt(S.reshape(-1, 2)).reshape(len(S), K + 1, -1), dtype=torch.float32); A = torch.tensor(A, dtype=torch.float32)
    E = mlp(O.shape[-1], d); opt = torch.optim.Adam(E.parameters(), lr=2e-3); n = len(O)
    for it in range(steps):
        idx = torch.randint(0, n, (256,)); o, a = O[idx], A[idx]
        z = E(o.reshape(-1, o.shape[-1])).reshape(256, K + 1, d); zh = z[:, 0]; ld = 0
        for k in range(K):
            zh = Pf(zh, a[:, k]); ld = ld + (zh - z[:, k + 1]).pow(2).mean()
        loss = ld / K + vic_reg(z.reshape(-1, d))
        if npair > 0:
            pi = torch.randint(0, npair, (min(32, npair),)); loss = loss + (E(Opair[pi]) - Zpair[pi]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return E
out = dict(src_consistency=src_cons)
for npair in [0, 1, 2, 5]:
    E = adapt(npair); raw, al = recov(E); out[f'pairs{npair}'] = dict(raw=raw, aligned=al)
    print(seed, npair, out[f'pairs{npair}'], round(time.time() - t0), flush=True)
json.dump(out, open(f'pilot_a2_s{seed}.json', 'w'), indent=1)
