"""Pilot A (kill test): can a NEW sensor be attached to a frozen pretrained latent predictor using only unlabelled
target transitions (dynamics consistency through the frozen predictor + unpaired marginal matching)?
Competitors: retrain from scratch on the same target transitions; few-pair regression; marginal matching only; dynamics only."""
import sys, json, time, numpy as np, torch
from common import *
seed = int(sys.argv[1]); NTR_LIST = [int(x) for x in sys.argv[2].split(',')]
d = 4; K = 5; NEP = 60
src = ObsMap(8, 1, 'tanh'); tgt = ObsMap(10, 7, 'sin')
res = {}
t0 = time.time()
E_src, P = train_jepa(src, d, 2000, K, seed, steps=2500)
for p in P.parameters(): p.requires_grad_(False)
for p in E_src.parameters(): p.requires_grad_(False)
Pf = lambda z, a: z + P(torch.cat([z, a], -1))
# reference latents of the source domain (unpaired: independent random states)
Sref = np.random.RandomState(seed + 5).uniform(-1, 1, (4000, 2))
Zref = E_src(torch.tensor(src(Sref), dtype=torch.float32))
def success(E, Pn, tag):
    return run_episodes(E, Pn, tgt, NEP, 123 + seed)
# oracle with source sensor (upper bound for frozen P) 
res['oracle_srcsensor'] = run_episodes(E_src, P, src, NEP, 123 + seed)
Spaired = np.random.RandomState(seed + 9).uniform(-1, 1, (2000, 2))
Zpair = E_src(torch.tensor(src(Spaired), dtype=torch.float32)); Opair = torch.tensor(tgt(Spaired), dtype=torch.float32)
def recovery(E):
    with torch.no_grad():
        Sx = np.random.RandomState(seed + 77).uniform(-1, 1, (1000, 2))
        z1 = E(torch.tensor(tgt(Sx), dtype=torch.float32)); z0 = E_src(torch.tensor(src(Sx), dtype=torch.float32))
        return float(((z1 - z0).pow(2).sum(-1).mean() / z0.var(0).sum()))
def adapt(mode, ntr, steps=2500, lam=1.0, Kuse=K, npair=0):
    torch.manual_seed(seed + 3)
    S, A = collect(ntr // K, K, seed + 2000 + ntr)
    O = torch.tensor(tgt(S.reshape(-1, 2)).reshape(len(S), K + 1, -1), dtype=torch.float32); A = torch.tensor(A, dtype=torch.float32)
    E = mlp(O.shape[-1], d); opt = torch.optim.Adam(E.parameters(), lr=2e-3)
    n = len(O)
    for it in range(steps):
        idx = torch.randint(0, n, (min(256, n),)); o, a = O[idx], A[idx]; b = len(idx)
        z = E(o.reshape(-1, o.shape[-1])).reshape(b, K + 1, d)
        loss = 0.0
        if mode in ('dyn+sw', 'dyn'):
            zh = z[:, 0]; ld = 0
            for k in range(Kuse):
                zh = Pf(zh, a[:, k]); ld = ld + (zh - z[:, k + 1]).pow(2).mean()
            loss = loss + ld / Kuse
        if mode in ('dyn+sw', 'sw'):
            ridx = torch.randint(0, len(Zref), (512,))
            loss = loss + lam * sliced_w2(z.reshape(-1, d), Zref[ridx])
        if mode == 'dyn':
            loss = loss + vic_reg(z.reshape(-1, d))
        if mode == 'pairs':
            pi = torch.randint(0, npair, (min(64, npair),))
            loss = (E(Opair[pi]) - Zpair[pi]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return E
for ntr in NTR_LIST:
    r = {}
    for mode, kw in [('dyn+sw', {}), ('dyn+sw_k1', dict(Kuse=1)), ('sw', {}), ('dyn', {}), ('pairs10', dict(npair=10)), ('pairs50', dict(npair=50)), ('pairs200', dict(npair=200))]:
        m = mode.split('_')[0] if mode.startswith('dyn+sw') else ('pairs' if mode.startswith('pairs') else mode)
        E = adapt(m, ntr, **kw)
        sr, fd = run_episodes(E, P, tgt, NEP, 123 + seed)
        r[mode] = dict(succ=sr, dist=fd, recov=recovery(E))
        print(seed, ntr, mode, r[mode], round(time.time() - t0), flush=True)
    # retrain from scratch on the same target transitions (also learns its own predictor)
    Er, Pr = train_jepa(tgt, d, ntr // K, K, seed + 50, steps=2500)
    sr, fd = run_episodes(Er, Pr, tgt, NEP, 123 + seed)
    r['retrain'] = dict(succ=sr, dist=fd); print(seed, ntr, 'retrain', r['retrain'], round(time.time() - t0), flush=True)
    res[str(ntr)] = r
json.dump(res, open(f'pilot_a_s{seed}.json', 'w'), indent=1)
