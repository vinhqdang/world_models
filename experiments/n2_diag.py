"""N2 diagnostic: structure of the learned noise map u -> z' of the energy-score predictors (effective noise dimension)."""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from selwm.jepa import JEPA
torch.manual_seed(0)
out = {}
for s in range(3):
    ck = torch.load(f'suite_hi/ckpt/es_s{s}.pt', map_location='cpu'); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); m.load_state_dict(ck['state']); m.eval()
    P = m.pred
    # states near the hazard / elsewhere, random actions
    n = 4000
    p = torch.stack([torch.rand(n) * 0.9 + 0.05, torch.rand(n) * 0.9 + 0.05], 1)
    z = torch.cat([p * 2 - 1, torch.zeros(n, 1)], 1)
    act = torch.rand(n, 2) * 2 - 1
    # Jacobian wrt u at u=0 and at random u: singular values of (3 x 8)
    sv, V = [], []
    for _ in range(3):
        u = torch.randn(n, 8, requires_grad=True)
        zp = P(z, act, u)[0]
        J = torch.stack([torch.autograd.grad(zp[:, d].sum(), u, retain_graph=True)[0] for d in range(3)], 1)   # n,3,8
        S = torch.linalg.svdvals(J); sv.append(S.mean(0))
        # direction consistency: top right singular vector (sign-free) projected outer product averaged
        _, _, Vh = torch.linalg.svd(J); V.append(Vh[:, 0, :])
    sv = torch.stack(sv).mean(0)
    v = V[0]; M2 = (v[:, :, None] * v[:, None, :]).mean(0)       # average outer product of top direction
    ev = torch.linalg.eigvalsh(M2).flip(0)
    # predictive law at a few fixed states: distribution of dy (should be bimodal +-0.09)
    zz = torch.tensor([[-0.2, -0.1, 0.0]]); aa = torch.tensor([[0.5, 0.0]])
    u = torch.randn(20000, 8)
    with torch.no_grad(): d = (P(zz.expand(20000, -1), aa.expand(20000, -1), u)[0] - zz)
    dy = d[:, 1] / 2                                              # in position units
    # regress dy on u linearly
    X = torch.cat([u, torch.ones(20000, 1)], 1); beta = torch.linalg.lstsq(X, dy[:, None]).solution[:, 0]
    r2 = 1 - ((X @ beta - dy) ** 2).mean() / dy.var()
    # nonlinear: first-principal-direction of u, then R2 of 1D function
    vdir = beta[:8] / beta[:8].norm(); t = u @ vdir
    q = torch.quantile(dy, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    # 1-D binned conditional mean explains how much variance
    bins = torch.bucketize(t, torch.quantile(t, torch.linspace(0, 1, 41)[1:-1]))
    cm = torch.zeros(40).index_add_(0, bins, dy) / torch.bincount(bins, minlength=40)
    r2_1d = 1 - ((cm[bins] - dy) ** 2).mean() / dy.var()
    # is x-component noisy?
    print(f'model {s}: mean singular values of dz/du {sv.numpy().round(4)}; top-dir outer-product eigs {ev.numpy().round(3)}')
    print(f'   at a fixed state: dy mean {dy.mean():.4f} sd {dy.std():.4f} quantiles {q.numpy().round(3)}; dx sd {d[:,0].std()/2:.4f}; linear-R2(u)={r2:.3f}, 1D-nonlinear-R2={r2_1d:.3f}')
    out[s] = dict(sv=sv.tolist(), ev=ev.tolist(), r2_lin=float(r2), r2_1d=float(r2_1d), dy_sd=float(dy.std()))
json.dump(out, open('results/n2/diag_noise_map.json', 'w'), indent=1)
