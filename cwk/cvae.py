"""CF-GPS with a learned conditional VAE structural causal model: y = dec(h, a, z), z ~ N(0, I). Abduction: z = encoder
mean q(z | h, a, y); counterfactual: decode under the new action with the same z. Trained on factual transitions only."""
import math

import numpy as np
import torch
import torch.nn as nn

from cwk.predictor import mlp


class CVAE(nn.Module):
    def __init__(self, dh, q, d, zdim=4, width=256, depth=2):
        super().__init__()
        self.zdim, self.d = zdim, d
        self.enc = mlp(dh + q + d, 2 * zdim, width, depth)
        self.dec = mlp(dh + q + zdim, d, width, depth)
        self.log_noise = nn.Parameter(torch.full((d,), -2.0))

    def encode(self, h, psi, y):
        mu, lv = self.enc(torch.cat([h, psi, y], -1)).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def decode(self, h, psi, z):
        return self.dec(torch.cat([h, psi, z], -1))


def fit_cvae(h, psi, y, ep, seed, steps=6000, lr=2e-3, batch=512, zdim=4, beta=1.0, threads=2, val_frac=0.1):
    """y: standardised factual next-state target on the kriging coordinates (B, d)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = np.unique(ep)
    val = set(rng.choice(ids, max(1, int(val_frac * len(ids))), replace=False).tolist())
    isv = np.array([e in val for e in ep])
    tr, va = np.where(~isv)[0], np.where(isv)[0]
    net = CVAE(h.shape[1], psi.shape[1], y.shape[1], zdim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def elbo(idx):
        mu, lv = net.encode(h[idx], psi[idx], y[idx])
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        rec = net.decode(h[idx], psi[idx], z)
        nv = torch.exp(net.log_noise)
        nll = 0.5 * (((rec - y[idx]) ** 2) / nv + net.log_noise).sum(-1)
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - lv - 1).sum(-1)
        return (nll + beta * kl).mean()

    best, best_state = math.inf, None
    for step in range(steps):
        idx = torch.as_tensor(rng.choice(tr, batch))
        loss = elbo(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                vl = float(elbo(torch.as_tensor(va[:4096])))
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def cf_transport(net, h, psi_src, psi_tgt, y_src_std):
    mu, _ = net.encode(h, psi_src, y_src_std)
    return net.decode(h, psi_tgt, mu)
