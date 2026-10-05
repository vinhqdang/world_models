"""Model-level diagnostics near the hazard: H-step imagined vs true (simulator, many noise draws) latent distribution and fall probability.
python experiments/d2_diag.py --ckpt X.pt --out results/d2/diag_X.json"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, LatentModel, FailureAnchor, observe

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', default=None)
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--H', type=int, default=10)
ap.add_argument('--S', type=int, default=200); ap.add_argument('--n_true', type=int, default=1000); ap.add_argument('--n_model', type=int, default=256)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
ck = torch.load(args.ckpt); a = ck['args']
m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=0.0, obs=a.get('obs', 'fixed')); m.load_state_dict(ck['state']); m.eval()
anchor = FailureAnchor(m.encode, args.variant, m.obs)
env = StochNav(args.variant, 1, 0)
rng = np.random.default_rng(args.seed)
S, H = args.S, args.H
# states: band within 0.14 above the pit edge (the region the planner must traverse), x within the pit span
s = np.stack([rng.uniform(0.2, 0.8, S), rng.uniform(0.30, 0.44, S)], 1)
goal = np.stack([rng.uniform(0.82, 0.94, S), rng.uniform(0.34, 0.46, S)], 1)
acts = np.zeros((S, H, 2)); aa = rng.uniform(-1, 1, (S, 2)); gd = np.arange(S) % 2 == 0
p = s.copy()
for t in range(H):                                  # half goal-directed, half correlated random, no feedback on the noise
    aa = 0.85 * aa + 0.55 * rng.normal(size=(S, 2))
    tgt = goal - p; tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    acts[:, t] = np.clip(np.where(gd[:, None], 0.7 * tgt + 0.6 * rng.normal(size=(S, 2)), aa), -1, 1)
    p = np.clip(p + env.step_size * acts[:, t], 0, 1)
# ground truth
nt = args.n_true
pt = np.repeat(s, nt, 0); ft = np.zeros(S * nt, bool); ever = np.zeros(S * nt, bool)
for t in range(H):
    at = np.repeat(acts[:, t], nt, 0)
    pn, fell = env.transition(pt, at, rng)
    pt = np.where(ft[:, None], pt, pn); ft = ft | fell
true_z = observe(torch.tensor(pt, dtype=torch.float32), torch.tensor(ft), args.variant, 'fixed').reshape(S, nt, -1)
true_fall = ft.reshape(S, nt).mean(1)
true_anchor = None
# imagined
lm = LatentModel(m, args.variant, stochastic=True, stage_w=0.0, anchor=anchor)
z0 = m.encode(observe(torch.tensor(s, dtype=torch.float32), None, args.variant, 'fixed'))
gen = torch.Generator().manual_seed(args.seed)
feat = lm.rollout(z0, torch.tensor(acts, dtype=torch.float32)[:, None], args.n_model, gen)    # (S,1,M,D+2)
img_z = feat[:, 0, :, :-2]; img_fail = feat[:, 0, :, -1].mean(1).numpy()
# anchor-detector applied to true final latents (any-step failure is replaced by terminal fall flag; fall is absorbing)
true_anchor_fall = (((true_z - anchor.z) ** 2).sum(-1) < anchor.tau).float().mean(1).numpy()

def edist(X, Y):                                   # energy distance (sqrt form) between sample sets X:(n,D), Y:(m,D)
    dxy = torch.cdist(X, Y).mean(); dxx = torch.cdist(X, X).sum() / (len(X) * (len(X) - 1)); dyy = torch.cdist(Y, Y).sum() / (len(Y) * (len(Y) - 1))
    return float(torch.sqrt(torch.clamp(2 * dxy - dxx - dyy, min=0)))
ed = np.array([edist(img_z[i], true_z[i, :400]) for i in range(S)])
ed_floor = np.array([edist(true_z[i, 400:600], true_z[i, :400]) for i in range(S)])   # sampling-noise floor (true vs true)
mean_err = (img_z.mean(1) - true_z.mean(1)).norm(dim=-1).numpy()
sd_ratio = (img_z.std(1) / true_z.std(1).clamp(min=1e-6)).mean(-1).numpy()
risky = true_fall > 0.02
err = np.abs(img_fail - true_fall)
res = dict(ckpt=args.ckpt, S=S, frac_risky=float(risky.mean()), edist_mean=float(ed.mean()), edist_floor=float(ed_floor.mean()),
           mean_err=float(mean_err.mean()), sd_ratio_mean=float(sd_ratio.mean()),
           true_fall_mean=float(true_fall.mean()), imag_fall_mean=float(img_fail.mean()),
           fall_abs_err=float(err.mean()), fall_abs_err_risky=float(err[risky].mean()) if risky.any() else None,
           fall_bias=float((img_fail - true_fall).mean()), fall_corr=float(np.corrcoef(img_fail, true_fall)[0, 1]),
           anchor_on_true_fall_mean=float(true_anchor_fall.mean()),
           fall_abs_err_vs_anchor_true=float(np.abs(img_fail - true_anchor_fall).mean()))
print(json.dumps(res))
if args.out:
    json.dump(res, open(args.out, 'w'))
