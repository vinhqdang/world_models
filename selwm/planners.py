import torch


def predict_cost(model, x0, acts, g):
    """x0:(B,2) acts:(B,N,H,2) g:(B,2) -> member-wise terminal cost (K,B,N)."""
    B, N, H, _ = acts.shape
    K = model.K
    x = x0[None, :, None, :].expand(K, B, N, 2).reshape(K, B * N, 2)
    a = acts.reshape(1, B * N, H, 2).expand(K, -1, -1, -1)
    for t in range(H):
        x = model(x, a[:, :, t])
    gg = g[None, :, None, :].expand(K, B, N, 2).reshape(K, B * N, 2)
    return ((x - gg) ** 2).sum(-1).reshape(K, B, N)


def cem(model, x0, g, H, N, iters=6, elite_frac=0.1, init_std=0.8, gen=None, score_fn=None):
    """Batched CEM over B problems. Returns best action seq (B,H,2) and its stats.

    score_fn(costs_KBN, acts) -> (B,N) score to minimise (default: ensemble-mean cost).
    """
    B = x0.shape[0]
    mu = torch.zeros(B, 1, H, 2)
    sd = torch.full((B, 1, H, 2), init_std)
    ne = max(4, int(N * elite_frac))
    for it in range(iters):
        acts = (mu + sd * torch.randn(B, N, H, 2, generator=gen)).clamp(-1, 1)
        with torch.no_grad():
            c = predict_cost(model, x0, acts, g)
        s = c.mean(0) if score_fn is None else score_fn(c, acts)
        idx = s.topk(ne, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, 2))
        mu = el.mean(1, keepdim=True)
        sd = el.std(1, keepdim=True).clamp_min(0.05)
    best = s.argmin(1)
    a_star = acts[torch.arange(B), best]
    return a_star, c[:, torch.arange(B), best].mean(0), c[:, torch.arange(B), best].std(0)
