"""Multi-step, trajectory-level energy-score training of the stochastic predictor (frozen identity encoder)."""
import torch
from .jepa import observe


def energy_score(zs, y):
    """zs:(M,B,D) samples, y:(B,D) target -> (B,) energy score  E||X-y|| - 0.5 E||X-X'||  (unbiased pair average)."""
    M = zs.shape[0]
    t1 = (zs - y).norm(dim=-1).mean(0)
    d = (zs.unsqueeze(0) - zs.unsqueeze(1)).pow(2).sum(-1).add(1e-12).sqrt()
    t2 = d.sum((0, 1)) / (M * (M - 1))
    return t1 - 0.5 * t2


def rollout_free(pred, z0, a, M, noise_dim):
    """Free-running rollout from the true z0 with fresh noise per step and the model's own outputs.
    z0:(B,D) a:(B,K,2) -> (M,B,K,D)"""
    B, K, _ = a.shape
    z = z0.unsqueeze(0).expand(M, B, -1)
    out = []
    for k in range(K):
        u = torch.randn(M, B, noise_dim, device=z0.device)
        z, _ = pred(z, a[:, k].unsqueeze(0).expand(M, -1, -1), u)
        out.append(z)
    return torch.stack(out, 2)


def window_loss(model, zt, a, M=8, joint_w=1.0, marg_w=1.0, one_w=0.0, step_w=None):
    """zt:(B,K+1,D) true latents, a:(B,K,2). Returns per-window loss (B,) and info.

    joint : ES of the concatenated trajectory z_1..z_K in R^{K*D} (scoring rule for the joint law, so it also
            constrains temporal dependence of the noise-driven errors).
    marg  : sum_k ES of the step-k marginal under the free-running rollout (proper for each marginal; the planner's
            cost uses per-step marginals), averaged over k.
    one   : teacher-forced one-step ES at every step (anchors the local dynamics, stabilises long rollouts)."""
    B, K, _ = a.shape
    D = zt.shape[-1]
    zs = rollout_free(model.pred, zt[:, 0], a, M, model.noise_dim)               # (M,B,K,D)
    loss = zt.new_zeros(B); info = {}
    if joint_w > 0:
        lj = energy_score(zs.reshape(M, B, K * D), zt[:, 1:].reshape(B, K * D)) / (K ** 0.5)   # /sqrt(K): per-step scale
        loss = loss + joint_w * lj; info['joint'] = float(lj.mean())
    if marg_w > 0:
        lm = torch.stack([energy_score(zs[:, :, k], zt[:, k + 1]) for k in range(K)], 1)       # (B,K)
        if step_w is not None:
            lm = lm * step_w
        lm = lm.mean(1)
        loss = loss + marg_w * lm; info['marg'] = float(lm.mean())
    if one_w > 0:
        z_in = zt[:, :-1].reshape(1, B * K, D).expand(M, -1, -1)
        u = torch.randn(M, B * K, model.noise_dim, device=zt.device)
        z1, _ = model.pred(z_in, a.reshape(1, B * K, -1).expand(M, -1, -1), u)
        lo = energy_score(z1, zt[:, 1:].reshape(B * K, D)).reshape(B, K).mean(1)
        loss = loss + one_w * lo; info['one'] = float(lo.mean())
    return loss, info


def encode_windows(P, F, variant, obs='fixed'):
    return observe(P, F, variant, obs)
