"""Variance reduction of noise-space importance sampling for failure probability on a trained energy-score model."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.tail_is import tilt_direction, fail_prob, ce_direction
torch.set_num_threads(2)
ck = torch.load(sys.argv[1], map_location='cpu'); a = ck['args']
m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')); m.load_state_dict(ck['state']); m.eval()
for p_ in m.parameters(): p_.requires_grad_(False)
variant = sys.argv[2] if len(sys.argv) > 2 else 'cliff_hi'
anchor = FailureAnchor(m.encode, variant, m.obs)
print('anchor acc %.3f tau %.3f' % (anchor.acc, anchor.tau))
rng = np.random.default_rng(0); H = 10; B = 400
p0 = np.stack([rng.uniform(0.25, 0.75, B), rng.uniform(0.30, 0.50, B)], 1)
acts = np.stack([np.ones((B, H)) * rng.uniform(0.2, 1.0, (B, 1)), rng.uniform(-0.9, 0.3, (B, H))], -1).astype(np.float32)
z0 = m.encode(observe(torch.tensor(p0, dtype=torch.float32), None, variant, m.obs)).detach(); acts = torch.tensor(acts)
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    truth = torch.cat([fail_prob(m.pred.__self__ if False else m, z0[i:i+100], acts[i:i+100], anchor, 4000, 'mc', gen=gen) for i in range(0, B, 100)]) if False else None
class Wrap:  # model interface used by tail_is: .pred(z,a,u), .noise_dim
    def __init__(s, jepa): s.pred, s.noise_dim = jepa.pred, jepa.noise_dim
W = Wrap(m)
truth = torch.cat([fail_prob(W, z0[i:i+100], acts[i:i+100], anchor, 6000, 'mc', gen=gen) for i in range(0, B, 100)])
for lo, hi in [(0.02, 0.5), (0.003, 0.08)]:
    keep = (truth > lo) & (truth < hi)
    if keep.sum() < 5: print('range', lo, hi, 'too few candidates', int(keep.sum())); continue
    print(f'--- candidates with {lo}<P<{hi}: n={int(keep.sum())}  mean P {truth[keep].mean():.4f}')
    for name, mkmu in [('grad eta=1', lambda: tilt_direction(W, z0, acts, anchor.z, anchor.tau, eta=1.0)),
                       ('ce Mp=16 damp=.5', lambda: ce_direction(W, z0, acts, anchor, 16, 0.5, gen)),
                       ('ce Mp=32 damp=.7', lambda: ce_direction(W, z0, acts, anchor, 32, 0.7, gen))]:
        out = {m_: [] for m_ in ['mc', 'is', 'mix']}
        for _ in range(40):
            mu = mkmu()
            for mode in out:
                # equal total particle budget: M=16 for mc/is/mix; CE pilots are extra (reported separately)
                out[mode].append(fail_prob(W, z0, acts, anchor, 16, mode, mu=mu, gen=gen))
        r = {k: (float(((torch.stack(v)[:, keep] - truth[keep]) ** 2).mean().sqrt()), float((torch.stack(v)[:, keep] == 0).float().mean())) for k, v in out.items()}
        rel = {k: r[k][0] / float(truth[keep].mean()) for k in r}
        print(f'{name:18s} RMSE  mc {r["mc"][0]:.4f}  is {r["is"][0]:.4f}  mix {r["mix"][0]:.4f} | relative  mc {rel["mc"]:.2f} is {rel["is"]:.2f} mix {rel["mix"]:.2f} | zero-est frac mc {r["mc"][1]:.2f} is {r["is"][1]:.2f} mix {r["mix"][1]:.2f}')
