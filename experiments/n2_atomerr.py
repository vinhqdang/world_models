"""N2: constants for the bias bound of atom-cloud propagation, measured on the trained es predictors.
  delta_q : W1 distance (position units, y-component) between the predictor's one-step law and its 2-atom quantiser (nodes +-c vbar, fitted weight)
  L       : W1-Lipschitz constant of the one-step law in z, estimated by the largest spectral norm of d z'/d z over random (z, a, u)
  mc_se   : standard deviation of the terminal lateral displacement divided by sqrt(8) (Monte-Carlo scale for M = 8)
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.jepa import JEPA
from selwm.n2_model import active_direction, atom_weight, _random_states

torch.manual_seed(0)
out = {}
for s in range(3):
    ck = torch.load(f'suite_hi/ckpt/es_s{s}.pt', map_location='cpu', weights_only=False); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); m.load_state_dict(ck['state']); m.eval()
    P = m.pred
    vbar = active_direction(m); p_fit = atom_weight(m, vbar)
    n = 2000
    z, act = _random_states(n, 31337)
    with torch.no_grad():
        U = torch.randn(n, 512, 8)
        y = P(z[:, None, :].expand(-1, 512, -1), act[:, None, :].expand(-1, 512, -1), U)[0][..., 1] / 2          # position units
        yp = P(z, act, 2 * vbar.expand(n, -1))[0][:, 1] / 2; ym = P(z, act, -2 * vbar.expand(n, -1))[0][:, 1] / 2
    ys, _ = y.sort(-1)
    # W1 between empirical law (512 equal masses) and two atoms (yp w.p. p_fit, ym w.p. 1-p_fit): quantile coupling
    k = int(round((1 - p_fit) * 512))
    atoms = torch.cat([ym[:, None].expand(-1, k), yp[:, None].expand(-1, 512 - k)], 1)
    w1 = (ys - atoms).abs().mean(-1)
    # also mean-removed version (the atom mean can be shifted to match the law's mean exactly)
    shift = (ys.mean(-1) - atoms.mean(-1))[:, None]
    w1c = (ys - (atoms + shift)).abs().mean(-1)
    # Lipschitz constant in z
    zz = z.clone().requires_grad_(True); u = torch.randn(n, 8)
    zp = P(zz, act, u)[0]
    J = torch.stack([torch.autograd.grad(zp[:, d].sum(), zz, retain_graph=True)[0] for d in range(3)], 1)           # n,3,3
    L = torch.linalg.matrix_norm(J, ord=2)
    out[s] = dict(p_fit=p_fit, delta_q_mean=float(w1.mean()), delta_q_p95=float(torch.quantile(w1, 0.95)), delta_q_meanshift=float(w1c.mean()),
                  L_mean=float(L.mean()), L_p95=float(torch.quantile(L, 0.95)), L_max=float(L.max()),
                  sd_one_step=float(ys.std(-1).mean()), H10_bound_position_units=float(w1.mean() * sum(float(L.mean()) ** k_ for k_ in range(10))),
                  mc_se_terminal_M8=float(ys.std(-1).mean() * np.sqrt(10) / np.sqrt(8)))
    print(s, {k_: round(v, 4) for k_, v in out[s].items()})
json.dump(out, open('results/n2/atomerr.json', 'w'), indent=1)
