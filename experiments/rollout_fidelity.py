"""Linear-probe diagnostic of imagined rollouts: decode latent -> (x, y) and compare with the simulator."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, observe, FailureAnchor
path = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ck = torch.load(path, map_location='cpu'); a = ck['args']
m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')); m.load_state_dict(ck['state']); m.eval()
rng = np.random.default_rng(0)
env = StochNav('cliff', 1, 0)
with torch.no_grad():
    P = rng.uniform(0, 1, (20000, 2)); P = P[~env._in(P, env.pit)]
    z = m.encode(observe(torch.tensor(P, dtype=torch.float32), None, 'cliff', m.obs)).numpy()
    X = np.c_[z, np.ones(len(z))]; W = np.linalg.lstsq(X, P, rcond=None)[0]
    pred = X @ W
    print('probe R2 x %.3f y %.3f | rmse %.4f' % tuple(list(1 - ((pred - P) ** 2).sum(0) / ((P - P.mean(0)) ** 2).sum(0)) + [np.sqrt(((pred - P) ** 2).mean())]))
    anc = FailureAnchor(m.encode, 'cliff', m.obs)
    print('failure anchor acc %.3f tau %.3f' % (anc.acc, anc.tau))
    # imagined vs true rollouts for random action sequences from random non-pit starts (away from the pit)
    n = 2000
    p0 = np.stack([rng.uniform(0.05, 0.2, n), rng.uniform(0.45, 0.7, n)], 1)
    acts = rng.uniform(-1, 1, (n, H, 2)) * np.array([1.0, 0.6])
    zi = m.encode(observe(torch.tensor(p0, dtype=torch.float32), None, 'cliff', m.obs))
    pt = p0.copy(); ptm = p0.copy()
    for t in range(H):
        at = torch.tensor(acts[:, t], dtype=torch.float32)
        zi = m.sample_next(zi, at)
        pt, _ = env.transition(pt, acts[:, t], rng)
        ptm = np.clip(ptm + 0.05 * np.clip(acts[:, t], -1, 1), 0, 1)               # noise-free reference
    pi = np.c_[zi.numpy(), np.ones(n)] @ W
    print('H=%d imagined-vs-true position  rmse(x) %.4f rmse(y) %.4f | std of true y %.4f | rmse vs noise-free mean %.4f' % (
        H, np.sqrt(((pi[:, 0] - pt[:, 0]) ** 2).mean()), np.sqrt(((pi[:, 1] - pt[:, 1]) ** 2).mean()), pt[:, 1].std(), np.sqrt(((pi - ptm) ** 2).mean())))
    print('imagined y std %.4f  vs true y std %.4f  (spread)' % (pi[:, 1].std(), pt[:, 1].std()))
