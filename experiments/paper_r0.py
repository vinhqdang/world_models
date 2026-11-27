"""R0: unit tests of the paper code (protocol: results/paper/PROTOCOL.md).
usage: python3 experiments/paper_r0.py sim | gates | eb SETTING SEED [THREADS] | all_quick
  sim    old cells reproduce the cached Stage 2 generator bit for bit (population, collection, development cohort)
  gates  gates are deterministic, shared across the counterfactual branches of a draw, prefix-stable, with the right rate
  eb     rc_eb_gmm / idt / glr_iso / quadrature reproduce the K0 numbers on the syn2 generator (within 0.3 R^2 points)"""
import sys, json, glob, time, os
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds

DATA = "/home/user/world_models/results/cwk/data"
OUT = "/home/user/world_models/results/paper"
os.makedirs(f"{OUT}/raw", exist_ok=True)


def say(*a):
    print(*a, flush=True)


def test_sim():
    # the Ball-in-Cup behaviour policy (pretrained DDPG, float32 torch inference) depends on the torch thread count at the
    # 1e-7 level; the cached Stage 2 data were generated with one thread (old and new code agree at any fixed thread count)
    import torch
    torch.set_num_threads(1)
    ok = True
    for name in ("E1_nu50_sat_iid50", "E1_nu50_deadzone_iid50", "E2_cup_iid50", "E3_fetch_iid50", "E1_nu50_gain_iid100"):
        w = cells.STAGE2[name]()
        ref = worlds.load(glob.glob(f"{DATA}/{name}_pop_512x64.npz")[0])
        mine = worlds.population(w, 4, 8)
        errs = {k: float(np.abs(mine[k] - ref[k][:4, :8 if k in ("truth", "eps") else None]).max())
                for k in ("truth", "eps", "past_eps", "sources", "actions", "past_actions", "posterior")}
        say(f"[sim] population {name}: max abs diff vs cached {errs}")
        ok &= all(v == 0.0 for v in errs.values())
        refc = worlds.load(glob.glob(f"{DATA}/{name}_col_16384x5.npz")[0])
        col = worlds.collect(w, 12)
        errc = {k: float(np.abs(col[k] - refc[k][:12]).max()) for k in ("states", "actions", "eps")}
        say(f"[sim] collection {name}: max abs diff vs cached {errc}")
        ok &= all(v == 0.0 for v in errc.values())
    name = "E1_nu50_gain_iid50"
    w = cells.STAGE2[name]()
    import dataclasses
    ref = worlds.load(glob.glob(f"{DATA}/{name}_dev_*.npz")[0])
    mine = worlds.population(dataclasses.replace(w, seed_offset=w.seed_offset + 5_000_000), 3, 8)
    e = float(np.abs(mine["truth"] - ref["truth"][:3, :8]).max())
    say(f"[sim] development cohort {name}: max abs diff {e}")
    ok &= e == 0.0
    # the new per-channel direction code path with the default direction equals the old path
    from cwk.sim import ChannelSimulator
    w2 = cells.STAGE2["E3_fetch_iid50"]()
    env = w2.make()
    a = np.array([0.1, -0.2, 0.3, 0.0])
    p_old = env.command_perturbation(a, np.array([0.5, 0.2]))
    w2.channels = [dict(w2.channels[0], dir=[1.0, -1.0, 1.0])]
    env2 = w2.make()
    p_new = env2.command_perturbation(a, np.array([0.5, 0.2]))
    say(f"[sim] explicit dir (1,-1,1) equals default direction: {np.abs(p_old - p_new).max()}")
    ok &= np.abs(p_old - p_new).max() == 0.0
    say("SIM TEST", "PASSED" if ok else "FAILED")
    return ok


def test_gates():
    ok = True
    w = cells.info_cell("E1o", "gated", 0.2, None, 1.0, sigma=0.6, tag="g_test", seed_offset=1_950_000)
    spec = w.sim_spec("population")
    E1, E2 = 16, 8
    p16 = worlds.population(w, 3, E1)
    p8 = worlds.population(w, 3, E2)
    pre = float(np.abs(p16["eps"][:, :E2] - p8["eps"]).max())
    say(f"[gates] draw prefix stability (16 vs 8 draws): {pre}")
    ok &= pre == 0.0
    nd = p16["eps"].shape[-1]
    for i in range(3):
        for j in range(nd):
            gate = worlds.gate_stream(spec["seed"], i, j, "query", E1) < 0.2
            ok &= bool(np.array_equal(gate, p16["eps"][i, :, j] != 0))
    say(f"[gates] eps != 0 pattern equals gate stream (3 parents x {nd} drivers): {ok}")
    rate = np.mean([(worlds.gate_stream(spec["seed"], i, 0, "query", 4000) < 0.2).mean() for i in range(20)])
    say(f"[gates] empirical gate rate {rate:.4f} (pi = 0.2)")
    ok &= abs(rate - 0.2) < 0.01
    # gates independent across drivers and parents
    a = worlds.gate_stream(spec["seed"], 0, 0, "query", 4000)
    b = worlds.gate_stream(spec["seed"], 0, 1, "query", 4000)
    c = worlds.gate_stream(spec["seed"], 1, 0, "query", 4000)
    say(f"[gates] correlation across drivers {np.corrcoef(a, b)[0, 1]:.3f}, across parents {np.corrcoef(a, c)[0, 1]:.3f}")
    ok &= abs(np.corrcoef(a, b)[0, 1]) < 0.06 and abs(np.corrcoef(a, c)[0, 1]) < 0.06
    # shared across counterfactual branches: independent replay of every (draw, action) branch with the stored eps
    env = w.make()
    worst = 0.0
    for i in range(3):
        for jj in (0, 5, 11):
            for k in range(4):
                env.reset(spec["seed"] + i)
                for t in range(2):
                    env.step(p16["past_actions"][i, t], p16["past_eps"][i, t])
                y = env.target(env.query(p16["actions"][i, k], p16["eps"][i, jj]))
                worst = max(worst, float(np.abs(y - p16["truth"][i, jj, k]).max()))
    say(f"[gates] replay of branches with the shared eps (3 parents x 3 draws x 4 actions): max diff {worst}")
    ok &= worst == 0.0
    # the same eps (hence the same gate) enters all 4 branches: zero pattern of eps independent of the action by construction;
    # check that a gated-off driver has no effect: draw with all drivers off equals the eps = 0 response
    idx = np.where((p16["eps"][0] == 0).all(-1))[0]
    if len(idx):
        env.reset(spec["seed"])
        for t in range(2):
            env.step(p16["past_actions"][0, t], p16["past_eps"][0, t])
        y0 = env.target(env.query(p16["actions"][0, 1], np.zeros(nd)))
        say(f"[gates] all-off draw equals the undisturbed branch: {np.abs(y0 - p16['truth'][0, idx[0], 1]).max()}")
        ok &= np.abs(y0 - p16["truth"][0, idx[0], 1]).max() == 0.0
    # laws: ranges
    for law in ("tanh", "gauss"):
        w3 = cells.info_cell("E3o", law, None, None, 1.0, sigma=0.6, seed_offset=1_950_000)
        p3 = worlds.population(w3, 2, 8)
        say(f"[gates] law {law}: eps range [{p3['eps'].min():.3f}, {p3['eps'].max():.3f}]")
        ok &= bool(np.abs(p3["eps"]).max() <= 1.0)
    say("GATES TEST", "PASSED" if ok else "FAILED")
    return ok


def test_channels():
    """Per-channel direction / axis keys and the rotate channel produce the intended command perturbations."""
    ok = True
    w = cells.misalign_cell("E2o", "tanh", True, sigma=0.6)
    env = w.make()
    a = np.array([0.1, -0.2])
    x = a / env.reference
    p = env.command_perturbation(a, np.array([0.5, 0.3]))
    s_r, s_o = w.channels[0]["scale"], w.channels[1]["scale"]
    want = np.array([-s_r * 0.5 * x[1], s_r * 0.5 * x[0]]) + s_o * 0.3 * env.direction
    err = float(np.abs(p - want).max())
    print(f"[channels] rotate + offset perturbation: max abs diff {err}", flush=True)
    ok &= err < 1e-12
    w = cells.info_cell("E3o", "tanh", None, 4, 1.0, sigma=0.6)
    env = w.make()
    a = np.array([0.2, -0.3, 0.1, 0.0])
    x = a / env.reference
    eps = np.array([1.0, 0.0, 0.0, 0.0])               # gain driver of actuator 0 only
    p = env.command_perturbation(a, eps)
    want = np.zeros(4)
    want[0] = w.channels[0]["scale"] * x[0]
    err = float(np.abs(p - want).max())
    print(f"[channels] gain on one actuator (axis 0): perturbation {np.round(p, 4)} (only coordinate 0, = s x_0): max abs diff {err}", flush=True)
    ok &= err < 1e-12
    eps = np.array([0.0, 0.0, 0.0, 1.0])               # offset driver, direction (1, -1, 1) on the declared axes
    p = env.command_perturbation(a, eps)
    err = float(np.abs(p - w.channels[3]["scale"] * np.array([1.0, -1.0, 1.0, 0.0])).max())
    print(f"[channels] offset direction (1,-1,1): max abs diff {err}", flush=True)
    ok &= err < 1e-12
    # the no-clipping bound for every driver at full amplitude
    for name in ("E1o_P", "E2o_P", "E3o_P", "E3o_rd", "MIS_E2o"):
        w = cells.ALL_PAPER[name]()
        env = w.make()
        worst = 0.0
        rng = np.random.default_rng(0)
        for _ in range(200):
            act = env.fix_actions(rng.uniform(-env.action_limit, env.action_limit, env.action_dim))
            e = rng.choice([-1.0, 1.0], env.n_drivers)
            nrm = env.action_gain * act + env.command_perturbation(env.action_gain * act, e)
            worst = max(worst, float(np.abs(nrm).max()))
        print(f"[channels] {name}: max |disturbed command| over 200 extreme draws {worst:.4f} (<= 1)", flush=True)
        ok &= worst <= 1.0 + 1e-12
    print("CHANNELS TEST", "PASSED" if ok else "FAILED", flush=True)
    return ok


def test_struct():
    """oracle.parent_posterior (exact over gate patterns, pivot quadrature) against brute-force self-normalised importance sampling."""
    from cwk import oracle as OR
    from scipy.special import ndtr
    rng = np.random.default_rng(1)
    J = np.array([[1.0, 0.3, 0.5], [-0.2, 0.8, -1.0]])
    pi = 0.4
    laws = [dict(kind="gated", pi=pi), dict(kind="gated", pi=pi), dict(kind="tanh")]
    post = [(0.0, 1.0), (0.3, 0.7), (0.0, 1.0)]

    def sample(n):
        a = (rng.random(n) < pi) * (2 * ndtr(rng.standard_normal(n)) - 1)
        e1 = (rng.random(n) < pi) * (2 * ndtr(post[1][0] + np.sqrt(post[1][1]) * rng.standard_normal(n)) - 1)
        return np.stack([a, e1, np.tanh(rng.standard_normal(n))], 1)
    ytrue = sample(12)
    y = ytrue @ J.T
    s = 0.05
    import warnings
    warnings.filterwarnings("ignore")
    eh = OR.parent_posterior(J, y, s, laws, post)
    P = sample(4_000_000)
    L = P @ J.T
    err = []
    for e in range(len(y)):
        w = np.exp(-0.5 * ((L - y[e]) ** 2).sum(1) / s ** 2)
        w /= w.sum()
        err.append(np.abs(w @ P - eh[e]).max())
    print(f"[struct] max abs difference to importance sampling per draw: {np.round(err, 3)} (median {np.median(err):.3f})", flush=True)
    ok = bool(np.median(err) < 0.02 and max(err) < 0.1)
    print("STRUCT TEST", "PASSED" if ok else "FAILED", flush=True)
    return ok


def test_eb(setting, seed, threads=2):
    import torch
    from cwk import eb, kernels as KR
    torch.set_num_threads(threads)
    t0 = time.time()
    D = eb.syn2_generate(setting, seed)
    n_t = len(D["rt"])
    psi = lambda z: np.c_[np.ones(len(z)), z]
    x, r, tr, va = D["x"], D["r"], D["tr"], D["va"]
    G0 = eb.hh_lift(psi(x[tr]), r[tr], 3, seed=seed, restarts=8)
    SST = ((D["tgt"] - D["tgt"].mean(0)) ** 2).sum()
    r2 = lambda p: float(100 * (1 - ((p - D["tgt"]) ** 2).sum() / SST))
    ref = json.load(open(f"/home/user/world_models/results/k0/raw/{setting}_s{seed}.json"))
    Gtr, Gva = eb.G_of(psi(x[tr]), G0), eb.G_of(psi(x[va]), G0)
    Gs, Gt = eb.G_of(psi(D["xt"]), G0), eb.G_of(psi(D["xt2"]), G0)
    rtr, rva, rt = (torch.as_tensor(a) for a in (r[tr], r[va], D["rt"]))
    out = {}
    mdl, v, info = eb.fit_rc_eb_gmm(Gtr, rtr, Gva, rva, Ks=(32,), restarts=4, steps=3000, seed=seed, log=say)
    xh = eb.posterior_xi(mdl, Gs, rt)
    out["rc_eb_gmm"] = r2(torch.einsum("bdr,br->bd", Gt, xh).numpy())
    out["gmm_val_nll"] = v
    say(f"[eb {setting} s{seed}] rc_eb_gmm {out['rc_eb_gmm']:.3f} (K0 {ref['rc_eb_gmm']:.3f}) val {v:.4f} (K0 {ref['gmm_val_nll']:.4f}) {time.time() - t0:.0f}s")
    idt, vi, vis = eb.fit_idt(torch.as_tensor(G0), torch.as_tensor(psi(x[tr])), rtr, torch.as_tensor(psi(x[va])), rva, K=4, steps=3000, restarts=4, seed=seed, log=say)
    with torch.no_grad():
        xh = eb.batched(idt.fwd, torch.as_tensor(psi(D["xt"])), rt)[1]
        Gi2 = idt.G(torch.as_tensor(psi(D["xt2"])))
        out["idt"] = r2(torch.einsum("bdr,br->bd", Gi2, xh).numpy())
        out["idt_val_nll"] = vi
        Gi_va, Gi_s, Gi_t = (idt.G(torch.as_tensor(psi(z))).numpy() for z in (x[va], D["xt"], D["xt2"]))
    say(f"[eb {setting} s{seed}] idt {out['idt']:.3f} (K0 {ref['idt']:.3f}) {time.time() - t0:.0f}s")
    gl = eb.glr_iso(idt, Gi_va, r[va], Gi_s, D["rt"])
    out["glr_by_lambda"] = {str(l): r2(np.einsum("bdr,br->bd", Gi_t, xh_)) for l, xh_ in gl["xhat"].items()}
    out["glr_lambda"] = gl["lam"]
    out["glr_iso"] = out["glr_by_lambda"][str(gl["lam"])]
    say(f"[eb {setting} s{seed}] glr_iso {out['glr_iso']:.3f} lambda {gl['lam']} (K0 {ref['glr_iso']:.3f}, lambda {ref['glr_lambda']})")
    if D["PI"] is not None:
        A = 1 / np.sqrt(D["PI"] / 3)
        pb = eb.bayes_gated(eb.syn2_G(D["xt"]), eb.syn2_G(D["xt2"]), D["rt"], D["PI"], D["noise"], A, 0.03)
    else:
        pb = eb.bayes_tanh(eb.syn2_G(D["xt"]), eb.syn2_G(D["xt2"]), D["rt"], D["noise"], 64)
    out["exact_bayes"] = r2(pb)
    say(f"[eb {setting} s{seed}] quadrature exact Bayes {out['exact_bayes']:.3f} (K0 {ref['exact_bayes']:.3f}) {time.time() - t0:.0f}s")
    cmp = {k: (out[k], ref[k], abs(out[k] - ref[k])) for k in ("rc_eb_gmm", "idt", "glr_iso", "exact_bayes")}
    ok = all(v[2] <= 0.3 for v in cmp.values())
    out.update(setting=setting, seed=seed, compare=cmp, ok=bool(ok), seconds=time.time() - t0)
    json.dump(out, open(f"{OUT}/raw/R0_eb_{setting}_s{seed}.json", "w"), indent=1)
    say(f"EB TEST {setting} s{seed}", "PASSED" if ok else "FAILED", {k: round(v[2], 3) for k, v in cmp.items()})
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sim":
        sys.exit(0 if test_sim() else 1)
    if cmd == "gates":
        sys.exit(0 if test_gates() else 1)
    if cmd == "channels":
        sys.exit(0 if test_channels() else 1)
    if cmd == "struct":
        sys.exit(0 if test_struct() else 1)
    if cmd == "eb":
        sys.exit(0 if test_eb(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2) else 1)
