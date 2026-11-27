"""R5: antisymmetric gauge (the HH lift is not identified when d >= 2 and the cross blocks have an antisymmetric part).
usage: python3 experiments/paper_r5.py synth [--seeds 3] | real [--seeds 5] | --smoke        (report: paper_report.py r5)
synth: port of experiments/cf_pilots/analysis/gauge.py (n 200k, test 40k, noise 0.05, uniform drivers, psi = (1, x1, x2)):
       worlds channel syn2 (3 drivers), rotation (I +- x1 J) xi, misalignment theta J x (r = 1), misalignment + isotropic offset
       (r = 3); arms oracle, hh_sym (symmetric closed form), hh_lift with R in {r, r+1, r+2} (8 restarts), rr.
real:  Ball-in-Cup misalignment cells MIS_E2o (rotate channel + offset) and MIS_E2o_nooff: arms hh_global, hh_binned_nug,
       rc_eb_gmm, oracle_kernel_lm (+ rr_learned, trained, oracle*, oracle_struct_bayes*)."""
from paper_common import *

TAG = "r5"
J = np.array([[1.0, 0.3], [-0.2, 0.8]])
Jr = np.array([[0, -1.0], [1, 0]])
NOISE = 0.05
psi = lambda z: np.c_[np.ones(len(z)), z]


def G_syn(x):
    P = np.zeros((len(x), 2, 3)); P[:, 0, 0] = 1; P[:, 1, 0] = -1; P[:, 0, 1] = x[:, 0]; P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)


def G_rot(x, sgn):
    return np.eye(2)[None] + sgn * x[:, :1, None] * Jr[None]


def G_mis(x, off):
    g = np.einsum("ij,bj->bi", Jr, x)[:, :, None]
    if off:
        g = np.concatenate([g, np.repeat(np.eye(2)[None], len(x), 0) * 0.5], -1)
    return g


WORLDS = {"channel_syn2": (G_syn, 3), "rotation_plusJ": (lambda x: G_rot(x, 1), 2), "rotation_minusJ": (lambda x: G_rot(x, -1), 2),
          "misalign_only": (lambda x: G_mis(x, 0), 1), "misalign_offset": (lambda x: G_mis(x, 1), 3)}


def run_world(name, seed, n=200000, nt=40000, restarts=8):
    from cwk import kernels as KR
    Gf, r_true = WORLDS[name]
    rng = np.random.default_rng(100 + seed)
    d = 2
    x = rng.uniform(-1, 1, (n, 2)); xi = rng.uniform(-np.sqrt(3), np.sqrt(3), (n, r_true))
    r = np.einsum("bdr,br->bd", Gf(x), xi) + NOISE * rng.standard_normal((n, d))
    xt, xt2 = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)); xit = rng.uniform(-np.sqrt(3), np.sqrt(3), (nt, r_true))
    rt = np.einsum("bdr,br->bd", Gf(xt), xit) + NOISE * rng.standard_normal((nt, d)); tgt = np.einsum("bdr,br->bd", Gf(xt2), xit)
    R2 = lambda p: round(float(100 * (1 - ((p - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum())), 2)

    def blupK(Kss, Kts, nug):
        return np.einsum("bij,bj->bi", Kts, np.linalg.solve(Kss + nug * np.eye(d), rt[..., None])[..., 0])
    Go, Go2 = Gf(xt), Gf(xt2)
    out = {"oracle": R2(blupK(Go @ Go.transpose(0, 2, 1), Go2 @ Go.transpose(0, 2, 1), NOISE ** 2)), "rr": R2(rt), "r_true": r_true}
    S = KR.hh_moments(psi(x), r)

    def Ksym(p1, p2):
        K = np.zeros((len(p1), d, d))
        for (j, k), M_ in S.items():
            K += (p1[:, j] * p2[:, k])[:, None, None] * M_ if j == k else 0.5 * (p1[:, j] * p2[:, k] + p1[:, k] * p2[:, j])[:, None, None] * M_
        return K
    Css = Ksym(psi(xt), psi(xt)); w, V = np.linalg.eigh(Css); Css = np.einsum("bij,bj,bkj->bik", V, np.maximum(w, 0), V)
    out["hh_sym"] = R2(blupK(Css, Ksym(psi(xt2), psi(xt)), NOISE ** 2))
    for R in (r_true, r_true + 1, r_true + 2):
        G, loss = KR.bm_lift(S, 3, d, R, restarts=restarts, seed=seed)
        Gs, Gt = np.einsum("bq,qdr->bdr", psi(xt), G), np.einsum("bq,qdr->bdr", psi(xt2), G)
        out[f"hh_lift_R{R}"] = R2(blupK(Gs @ Gs.transpose(0, 2, 1), Gt @ Gs.transpose(0, 2, 1), NOISE ** 2))
        out[f"hh_lift_R{R}_fit"] = float(loss)
    return out


REAL = ["MIS_E2o", "MIS_E2o_nooff"]
REAL_ARMS = ["hh_global", "hh_binned_nug", "rc_eb_gmm", "oracle_kernel_lm"]

if __name__ == "__main__":
    a = cli(sys.argv[1:], default_seeds=3)
    log = lambda *x, **k: print(*x, flush=True)
    if a.smoke:
        print(run_world("rotation_plusJ", 0, n=20000, nt=4000, restarts=2))
        w, cell, tmp = smoke_cell("MIS_E2o", ["hh_global", "rc_eb_gmm"], TAG)
        print("R5 smoke ok", tmp)
        sys.exit(0)
    cmd = a.cells[0] if a.cells else "synth"
    if cmd == "synth":
        res = {n: [run_world(n, s) for s in range(a.seeds)] for n in WORLDS}
        json.dump(res, open(f"{PAPER}/R5_synth.json", "w"), indent=1)
        print(json.dumps({n: res[n][0] for n in res}, indent=1))
    elif cmd == "real":
        for n in REAL:
            run_cell(n, REAL_ARMS, range(a.seeds if a.seeds != 3 else 5), TAG, log=log)
