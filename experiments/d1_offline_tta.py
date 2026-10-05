"""Offline proxy for online TTA: adapt a trained model on true-simulator transitions at wind W1 with the TTA machinery
(last layer or all params, source replay, steps) and report predictive spread / energy score at W0 and W1."""
import argparse, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.jepa import JEPA, observe
from selwm.stochnav import StochNav
ap = argparse.ArgumentParser()
ap.add_argument('--kind', default='es'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tta_params', default='out'); ap.add_argument('--tta_lr', type=float, default=1e-3); ap.add_argument('--tta_replay', type=float, default=0.5)
ap.add_argument('--tta_batch', type=int, default=256); ap.add_argument('--iters', type=int, default=320); ap.add_argument('--steps', type=int, default=2)
ap.add_argument('--bufsteps', type=int, default=100)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
ck = torch.load(f'suite_hi/ckpt/{a.kind}_s{a.seed}.pt', map_location='cpu'); ar = ck['args']
m = JEPA(ar['kind'], ar['dim'], noise_dim=ar.get('noise_dim', 8), sigreg_weight=ar['sigreg'], obs=ar.get('obs', 'pixel')); m.load_state_dict(ck['state']); m.eval()
d = np.load('runs/data_cliff_hi_300000_e0.25.npz'); n = len(d['P'])
sel = np.random.default_rng(0).choice(n, 60000, replace=False)
S = [torch.tensor(d[k][sel], dtype=torch.float32 if k in ('P', 'A', 'P2') else torch.bool) for k in ['P', 'F', 'A', 'P2', 'F2']]
def sbatch(k):
    i = torch.randint(0, len(S[0]), (k,)); return observe(S[0][i], S[1][i], 'cliff_hi', m.obs), S[2][i], observe(S[3][i], S[4][i], 'cliff_hi', m.obs)
env = StochNav('cliff_hi', 1, 123)
def new_batch(k, wind):
    i = torch.randint(0, len(S[0]), (k,)); P, A = S[0][i].numpy().astype(np.float64), S[2][i].numpy()
    F = S[1][i].numpy(); env.wind_base = wind
    pn, fell = env.transition(P, A, np.random.default_rng(np.random.randint(1 << 30)))
    pn = np.where(F[:, None], P, pn)
    return observe(torch.tensor(P, dtype=torch.float32), torch.tensor(F), 'cliff_hi', m.obs), torch.tensor(A, dtype=torch.float32), observe(torch.tensor(pn, dtype=torch.float32), torch.tensor(F | fell), 'cliff_hi', m.obs)
test = {w: new_batch(4000, w) for w in (0.09, 0.13)}
@torch.no_grad()
def evaluate(tag):
    out = []
    for w, (o0, ac, o1) in test.items():
        pl = m.loss(o0, ac, o1, M=16)[1]['pred']
        if m.kind == 'es':
            u = torch.randn(64, len(o0), m.noise_dim); zs = m.pred(o0.expand(64, -1, -1), ac.expand(64, -1, -1), u)[0]
            sd = zs[..., 1].std(0)
        elif m.kind == 'gauss':
            mu, ls = m.pred(o0, ac); sd = ls[..., 1].exp()
        else: sd = torch.zeros(len(o0))
        out.append(f'wind{w}: loss {pl:.4f} pred-sd(y)/true {float(sd.mean())/ (w*2/2):.2f}*w (sd {float(sd.mean()):.3f}, true sd {w/2:.3f}x2 scaled)')
    print(tag, ' | '.join(out), flush=True)
for p_ in m.pred.parameters(): p_.requires_grad_(False)
params = list(m.pred.parameters()) if a.tta_params == 'all' else list(m.pred.out.parameters())
for p_ in params: p_.requires_grad_(True)
opt = torch.optim.Adam(params, lr=a.tta_lr)
evaluate('before adaptation')
buf = []
t0 = time.time()
for it in range(a.iters):
    buf.append(new_batch(16, 0.13)); buf[:] = buf[-a.bufsteps:]
    B = [torch.cat([b[j] for b in buf]) for j in range(3)]
    ns = int(round(a.tta_replay * a.tta_batch))
    for _ in range(a.steps):
        idx = torch.randint(0, len(B[0]), (a.tta_batch - ns,)); O0, AC, O1 = B[0][idx], B[1][idx], B[2][idx]
        if ns: b = sbatch(ns); O0, AC, O1 = torch.cat([O0, b[0]]), torch.cat([AC, b[1]]), torch.cat([O1, b[2]])
        loss, _ = m.loss(O0, AC, O1, M=8); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if (it + 1) in (20, 60, 120, 320, 640): evaluate(f'after {it+1} env steps')
print('sec', time.time() - t0)
