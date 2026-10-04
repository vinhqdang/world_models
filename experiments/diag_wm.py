"""Predictive-distribution diagnostics near the cliff edge.

For states close to the pit we draw K ground-truth next frames from the simulator (random wind), encode them, and
compare with K predictive samples of the model: energy distance, mean-embedding error, and the predicted fall
probability (fraction of samples closer to the 'game over' embedding than to the nearest non-fall embedding).
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav, render
from selwm.jepa import JEPA

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', nargs='+', required=True); ap.add_argument('--K', type=int, default=256)
ap.add_argument('--n_states', type=int, default=200); ap.add_argument('--out', default=None)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
dev = torch.device(args.device)


def edist(X, Y):
    """Energy distance between sample sets X (K,D), Y (K,D)."""
    xy = torch.cdist(X, Y).mean()
    xx = torch.cdist(X, X).sum() / (len(X) * (len(X) - 1))
    yy = torch.cdist(Y, Y).sum() / (len(Y) * (len(Y) - 1))
    return float(2 * xy - xx - yy)


rows = []
for ck_path in args.ckpt:
    ck = torch.load(ck_path, map_location=dev); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg']).to(dev); m.load_state_dict(ck['state']); m.eval()
    env = StochNav('cliff', 1, 0)
    rng = np.random.default_rng(7)
    # states hugging the pit edge, actions pushing toward the pit / along the edge
    px = rng.uniform(0.3, 0.7, args.n_states); py = rng.uniform(0.30, 0.40, args.n_states)
    acts = np.stack([rng.uniform(-0.2, 1.0, args.n_states), rng.uniform(-1.0, 0.2, args.n_states)], 1)
    ed, mean_err, p_true, p_model, edist_det = [], [], [], [], []
    with torch.no_grad():
        z_black = m.encode(render(torch.zeros(1, 2, device=dev), 'cliff', torch.ones(1, dtype=torch.bool, device=dev)))[0]
        for i in range(args.n_states):
            p = np.repeat([[px[i], py[i]]], args.K, 0); act = np.repeat([acts[i]], args.K, 0)
            pn, fell = env.transition(p, act, rng)
            x1 = render(torch.tensor(pn, dtype=torch.float32, device=dev), 'cliff', torch.tensor(fell, device=dev))
            zt = m.encode(x1)                                                       # true next-latent samples
            z0 = m.encode(render(torch.tensor(p[:1], dtype=torch.float32, device=dev), 'cliff'))
            zs = m.sample_next(z0.expand(args.K, -1), torch.tensor(act, dtype=torch.float32, device=dev))
            ed.append(edist(zs, zt))
            mean_err.append(float((zs.mean(0) - zt.mean(0)).norm()))
            d_b_t = (zt - z_black).norm(dim=-1); d_b_s = (zs - z_black).norm(dim=-1)
            # fall = closer to the black-frame embedding than to the mean of the non-fall true samples
            ref = zt[~torch.tensor(fell, device=dev)].mean(0) if (~fell).any() else zt.mean(0)
            p_true.append(float(fell.mean()))
            p_model.append(float(((zs - z_black).norm(dim=-1) < (zs - ref).norm(dim=-1)).float().mean()))
    p_true, p_model = np.array(p_true), np.array(p_model)
    r = dict(ckpt=ck_path, kind=a['kind'], energy_dist=float(np.mean(ed)), mean_emb_err=float(np.mean(mean_err)),
             p_fall_true=float(p_true.mean()), p_fall_model=float(p_model.mean()),
             p_fall_mae=float(np.abs(p_true - p_model).mean()),
             p_fall_corr=float(np.corrcoef(p_true, p_model)[0, 1]) if p_model.std() > 0 else float('nan'))
    rows.append(r); print(r, flush=True)
if args.out:
    json.dump(rows, open(args.out, 'w'), indent=1)
