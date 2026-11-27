"""R4: sign ambiguity (column-zero sign non-identifiability).
usage: python3 experiments/paper_r4.py synth [--seeds 5] | real [--seeds 5] | --smoke        (report: paper_report.py r4)
synth: d = 1, y = xi_0 + s(a) xi_1 + n (n sd 0.05, a ~ U(-1, 1), 200k factual, 40k test), s in {a, |a|, a|a|, sign(a) exp(-1/a^2)},
       xi_1 symmetric (uniform) or skewed (standardised Gamma(4)), xi_0 uniform; bases (1,a), (1,|a|), (1,a,|a|), (1,a,a^2,a^3) with
       cond(Phi^T Phi) and a channel-free MLP G(a) head; held-out factual Gaussian NLL, R^2 on all / cross-zero / same-side queries.
real:  hand_affine100 (authors' pure-gain cell) and E1_nu50_gain_iid100 with psi (1,x) vs (1,|x|) (arms hh_basis_x, hh_basis_abs;
       factual NLL in the diagnostics, per-(parent, action) losses for the cross-zero split)."""
from paper_common import *

TAG = "r4"
SHAPES = {"a": lambda a: a, "abs": np.abs, "a|a|": lambda a: a * np.abs(a),
          "flat": lambda a: np.sign(a) * np.exp(-1.0 / np.maximum(a ** 2, 1e-12))}
BASES = {"(1,a)": lambda z: np.c_[np.ones_like(z), z], "(1,|a|)": lambda z: np.c_[np.ones_like(z), np.abs(z)],
         "(1,a,|a|)": lambda z: np.c_[np.ones_like(z), z, np.abs(z)], "(1,a,a2,a3)": lambda z: np.c_[np.ones_like(z), z, z ** 2, z ** 3]}
NOISE = 0.05


def draw_xi(rng, m, skew):
    xi0 = rng.uniform(-np.sqrt(3), np.sqrt(3), m)
    if skew:
        k = 4.0
        xi1 = (rng.gamma(k, 1.0, m) - k) / np.sqrt(k)
    else:
        xi1 = rng.uniform(-np.sqrt(3), np.sqrt(3), m)
    return np.stack([xi0, xi1], 1)


def mlp_head(a, y, a_va, y_va, seed, steps=3000):
    """Channel-free G(a) in R^{1 x 2}: MLP trained by the factual Gaussian NLL with N = noise^2 (second moments only)."""
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2))
    A, Y = torch.as_tensor(a[:, None], dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    Av, Yv = torch.as_tensor(a_va[:, None], dtype=torch.float32), torch.as_tensor(y_va, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    nl = lambda A_, Y_: (0.5 * (Y_ ** 2 / ((net(A_) ** 2).sum(1) + NOISE ** 2) + torch.log((net(A_) ** 2).sum(1) + NOISE ** 2))).mean()
    rg = np.random.default_rng(seed)
    best = (1e9, None)
    for st in range(steps):
        i = torch.as_tensor(rg.choice(len(A), 1024))
        l = nl(A[i], Y[i])
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
        if (st + 1) % 500 == 0 or st + 1 == steps:
            with torch.no_grad():
                v = float(nl(Av, Yv))
            if v < best[0]:
                best = (v, {k: p.clone() for k, p in net.state_dict().items()})
    net.load_state_dict(best[1])
    return lambda z: net(torch.as_tensor(z[:, None], dtype=torch.float32)).detach().numpy().astype(np.float64)


def synth_one(shape, skew, seed, n=200000, nt=40000, restarts=8, mlp_steps=3000):
    from cwk import kernels as KR
    rng = np.random.default_rng(1000 * seed + 17)
    s = SHAPES[shape]
    world = lambda a, xi: xi[:, 0] + s(a) * xi[:, 1]
    a = rng.uniform(-1, 1, n); xi = draw_xi(rng, n, skew); y = world(a, xi) + NOISE * rng.standard_normal(n)
    va = rng.permutation(n)[: n // 10]; tr = np.setdiff1d(np.arange(n), va)
    at, at2 = rng.uniform(-1, 1, nt), rng.uniform(-1, 1, nt); xit = draw_xi(rng, nt, skew)
    yt = world(at, xit) + NOISE * rng.standard_normal(nt); tgt = world(at2, xit)
    cross = np.sign(at) != np.sign(at2)
    SST = lambda m: ((tgt[m] - tgt.mean()) ** 2).sum()
    out = {}

    def evaluate(name, Gs_fn, Gt_fn, nll):
        Gs, Gt = Gs_fn(at), Gt_fn(at2)
        A = Gs @ Gs.T if False else None
        Gs3, Gt3 = Gs[:, None, :], Gt[:, None, :]
        Amat = Gs3.transpose(0, 2, 1) @ Gs3 + NOISE ** 2 * np.eye(Gs.shape[1])
        w = np.linalg.solve(Amat, (Gs3.transpose(0, 2, 1) * yt[:, None, None]).sum(-1)[..., None])[..., 0]
        p = (Gt * w).sum(1)
        r2 = lambda m: float(100 * (1 - ((p[m] - tgt[m]) ** 2).sum() / SST(m)))
        allm = np.ones(nt, bool)
        out[name] = dict(r2=r2(allm), r2_cross=r2(cross), r2_same=r2(~cross), nll=nll)
    for bn, bf in BASES.items():
        P = bf(a)
        Phi, _ = KR.product_basis(P)
        cond = float(np.linalg.cond(Phi.T @ Phi))
        S = KR.hh_moments(P[tr], y[tr, None])
        G, _ = KR.bm_lift(S, P.shape[1], 1, 2, restarts=restarts, seed=seed)
        Gf = lambda z, bf=bf, G=G: np.einsum("bq,qdr->br", bf(z), G)
        Gva = Gf(a[va])
        nll = float(np.mean(0.5 * (y[va] ** 2 / ((Gva ** 2).sum(1) + NOISE ** 2) + np.log((Gva ** 2).sum(1) + NOISE ** 2) + np.log(2 * np.pi))))
        evaluate(bn, Gf, Gf, nll)
        out[bn]["cond"] = cond
    hd = mlp_head(a[tr], y[tr], a[va], y[va], seed, steps=mlp_steps)
    Gva = hd(a[va])
    nll = float(np.mean(0.5 * (y[va] ** 2 / ((Gva ** 2).sum(1) + NOISE ** 2) + np.log((Gva ** 2).sum(1) + NOISE ** 2) + np.log(2 * np.pi))))
    evaluate("mlp_G", hd, hd, nll)
    # oracle (true kernel) and residual reuse
    Gs = np.stack([np.ones(nt), s(at)], 1); Gt = np.stack([np.ones(nt), s(at2)], 1)
    Amat = Gs[:, :, None] * Gs[:, None, :] + NOISE ** 2 * np.eye(2)
    w = np.linalg.solve(Amat, (Gs * yt[:, None])[..., None])[..., 0]
    p = (Gt * w).sum(1)
    out["oracle"] = dict(r2=float(100 * (1 - ((p - tgt) ** 2).sum() / SST(np.ones(nt, bool)))))
    out["rr"] = dict(r2=float(100 * (1 - ((yt - tgt) ** 2).sum() / SST(np.ones(nt, bool)))))
    return out


def run_synth(seeds, **kw):
    res = {}
    for shape in SHAPES:
        for skew in (0, 1):
            res[f"{shape}|{'skew' if skew else 'sym'}"] = [synth_one(shape, skew, s, **kw) for s in range(seeds)]
            print(shape, skew, "done", flush=True)
    json.dump(res, open(f"{PAPER}/R4_synth.json", "w"))
    return res


REAL = ["hand_affine100", "E1_nu50_gain_iid100"]
REAL_ARMS = ["hh_basis_x", "hh_basis_abs", "oracle_kernel_lm"]

if __name__ == "__main__":
    a = cli(sys.argv[1:])
    log = lambda *x, **k: print(*x, flush=True)
    if a.smoke:
        r = synth_one("a", 0, 0, n=20000, nt=4000, restarts=2, mlp_steps=300)
        print({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in r.items()})
        w, cell, tmp = smoke_cell("E1_nu50_gain_iid100", ["hh_basis_x", "hh_basis_abs"], TAG, extra_cfg=dict(save_pk=True))
        print("R4 smoke ok", tmp)
        sys.exit(0)
    cmd = a.cells[0] if a.cells else "synth"
    if cmd == "synth":
        run_synth(a.seeds)
    elif cmd == "real":
        for n in REAL:
            run_cell(n, REAL_ARMS, range(a.seeds), TAG, extra_cfg=dict(save_pk=True), log=log)
