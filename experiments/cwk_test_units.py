"""Unit tests of cwk.kernels on analytic worlds with a known kernel. Run: python3 experiments/cwk_test_units.py"""
import sys, math
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import analytic, kernels as KR, predictor as P, metrics as M


def test_woodbury_matches_dense():
    torch.manual_seed(0)
    B, d, R = 5, 6, 3
    G, r = torch.randn(B, d, R), torch.randn(B, d)
    n = torch.rand(d) + 0.1
    got = KR.nll_factor(G, r, n)
    for b in range(B):
        S = G[b].double() @ G[b].double().T + torch.diag(n.double())
        ref = -torch.distributions.MultivariateNormal(torch.zeros(d, dtype=torch.float64), S).log_prob(r[b].double())
        assert abs(float(got[b] - ref)) < 1e-8, (got[b], ref)
    # pair likelihood
    G1, G2, r1, r2 = torch.randn(B, d, R), torch.randn(B, d, R), torch.randn(B, d), torch.randn(B, d)
    kappa = torch.tensor(0.4)
    got = KR.nll_pair(G1, G2, r1, r2, n, kappa)
    for b in range(B):
        K11 = G1[b].double() @ G1[b].double().T + torch.diag(n.double())
        K22 = G2[b].double() @ G2[b].double().T + torch.diag(n.double())
        K12 = 0.4 * G1[b].double() @ G2[b].double().T
        S = torch.cat([torch.cat([K11, K12], 1), torch.cat([K12.T, K22], 1)], 0)
        z = torch.cat([r1[b], r2[b]]).double()
        ref = -torch.distributions.MultivariateNormal(torch.zeros(2 * d, dtype=torch.float64), S).log_prob(z)
        assert abs(float(got[b] - ref)) < 1e-8, (got[b], ref)
    print("woodbury / pair likelihood match dense Gaussian: ok")


def test_krige_matches_blup():
    """G' (G^T G + tau)^-1 G^T r equals K(a', a) [K(a, a) + tau I]^-1 r."""
    torch.manual_seed(1)
    d, R = 5, 3
    Gs, Gt, r = torch.randn(1, d, R).double(), torch.randn(1, d, R).double(), torch.randn(1, d).double()
    Kss, Kts = Gs[0] @ Gs[0].T, Gt[0] @ Gs[0].T
    ref = Kts @ torch.linalg.solve(Kss + 1e-3 * torch.eye(d, dtype=torch.float64), r[0])
    A = Gs[0].T @ Gs[0] + 1e-3 * torch.eye(R, dtype=torch.float64)
    got = Gt[0] @ torch.linalg.solve(A, Gs[0].T @ r[0])
    assert torch.allclose(ref, got, atol=1e-9), (ref, got)
    print("push-through kriging identity: ok")


def exact_residuals(w, col):
    H, a, y, s, ep, st = P.flatten_transitions(col)
    mu = np.concatenate([w.mean_next(s[:, :w.d], s[:, w.d], a), np.zeros((len(a), 1))], 1)
    sc = P.Scaler(H, a, y, s)
    kd = KR.prepare(H, a, y, s, ep, st, mu, list(range(w.d)), w.axes, 1.0, sc, col["actions"].shape[1])
    return kd, s, a


def kernel_error(w, head, kd, s, a, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(kd.n, n, replace=False)
    a2 = rng.uniform(-1, 1, (n, w.d))
    psi2 = torch.as_tensor(KR.psi_features(a2, w.axes, 1.0)[0], dtype=torch.float32)
    G1 = head.G(kd.h[idx], kd.psi[idx]).double()
    G2 = head.G(kd.h[idx], psi2).double()
    Kest = (G1 @ G2.transpose(1, 2)).numpy() * (kd.sigma[:, None] * kd.sigma[None, :])
    Ktrue = w.true_kernel(s[idx, w.d], a[idx], a2)
    return float(np.linalg.norm(Kest - Ktrue) / np.linalg.norm(Ktrue))


def test_kernel_recovery(steps=3000):
    out = {}
    for name, w, kinds in [("rank-2, iid", analytic.two_driver(0.6, d=2, iid=1.0, seed=3), ["polar"]),
                           ("rank-2, iid 0.5", analytic.two_driver(0.6, d=2, iid=0.5, seed=4), ["polar", "joint", "echo"]),
                           ("rank-1 gain, iid 0.5", analytic.dial(1.0, d=2, iid=0.5, seed=5), ["polar", "joint"])]:
        col = w.collect(8000, T=5)
        kd, s, a = exact_residuals(w, col)
        for kind in kinds:
            head = KR.fit_head(kd, kind, 0, steps=steps)
            err = kernel_error(w, head, kd, s, a)
            out[(name, kind)] = err
            print(f"kernel recovery {name:22s} {kind:6s} rel. Frobenius error {err:.3f}  kappa {float(head.kappa):.3f}", flush=True)
    return out


def test_closed_form():
    w = analytic.Analytic([dict(kind="offset", scale=0.4), dict(kind="gain", scale=0.5)], d=2, gain=0.3, iid=1.0, seed=9, ctx_gain=0.0)
    col = w.collect(8000, T=5)
    kd, s, a = exact_residuals(w, col)
    S = KR.hh_moments(kd.psi.numpy().astype(float), kd.r.numpy().astype(float))
    G, loss = KR.bm_lift(S, kd.q, kd.d, 4, restarts=8, seed=0)

    class H:  # head-like wrapper for kernel_error
        def G(self, h, psi):
            return torch.einsum("bq,qdr->bdr", psi.double(), torch.as_tensor(G))
    err = kernel_error(w, H(), kd, s, a)
    print(f"context-free closed form (moment regression + Burer-Monteiro lift): rel error {err:.3f}, lift loss {loss:.2e}")
    return err


if __name__ == "__main__":
    test_woodbury_matches_dense()
    test_krige_matches_blup()
    e = test_closed_form()
    assert e < 0.2, e
    res = test_kernel_recovery()
    # polarisation-only identifies K up to the antisymmetric gauge in general, but the bi-affine worlds here are identified
    assert res[("rank-2, iid", "polar")] < 0.35
    print("ALL UNIT TESTS PASSED")
