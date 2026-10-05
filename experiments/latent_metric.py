"""How faithful is squared latent distance to true goal distance? (Spearman, overall and within distance bins)."""
import sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.')
from scipy.stats import spearmanr
from selwm.stochnav import render
from selwm.jepa import JEPA
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', nargs='+', required=True); ap.add_argument('--variant', default='cliff')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); args = ap.parse_args()
dev = torch.device(args.device)
for path in args.ckpt:
    ck = torch.load(path, map_location=dev); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg']).to(dev); m.load_state_dict(ck['state']); m.eval()
    rng = np.random.default_rng(0); out = []
    for gx, gy in [(0.88, 0.40), (0.85, 0.45), (0.9, 0.36)]:
        g = np.array([[gx, gy]])
        P = rng.uniform(0, 1, (4000, 2)); P = P[~((P[:, 0] > 0.25) & (P[:, 0] < 0.75) & (P[:, 1] < 0.30))] if args.variant == 'cliff' else P
        with torch.no_grad():
            zg = m.encode(render(torch.tensor(g, dtype=torch.float32, device=dev), args.variant))
            z = m.encode(render(torch.tensor(P, dtype=torch.float32, device=dev), args.variant))
        dl = ((z - zg) ** 2).sum(-1).cpu().numpy(); dt = np.linalg.norm(P - g, axis=1)
        out.append([spearmanr(dl, dt)[0]] + [spearmanr(dl[(dt >= lo) & (dt < hi)], dt[(dt >= lo) & (dt < hi)])[0] for lo, hi in [(0, .1), (.1, .2), (.2, .4), (.4, .8)]])
    o = np.mean(out, 0)
    print(f'{path:28s} dim={a["dim"]:3d} kind={a["kind"]:5s} sigreg={a["sigreg"]}  spearman overall {o[0]:.3f} | bins [0,.1) {o[1]:.2f} [.1,.2) {o[2]:.2f} [.2,.4) {o[3]:.2f} [.4,.8) {o[4]:.2f}', flush=True)
