"""Pilot: compounding-error fixes on embedded chaotic / limit-cycle systems.
usage: p2_run.py <system> <tune|test> <seed_list comma> [out.jsonl]"""
import sys, time, json, torch, numpy as np
sys.path.insert(0, '/home/user/world_models')
torch.set_num_threads(1)
from selwm.p2_dyn import *
from selwm.p2_train import *

name, phase, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
out = sys.argv[4] if len(sys.argv) > 4 else f"/home/user/world_models/results/p2/{name}_{phase}.jsonl"
STEPS, BS, LR, NTR = 4000, 1024, 2e-3, 100
role = "val" if phase == "tune" else "test"
GRID = [0.01, 0.03, 0.1, 0.3]


def log(rec):
    with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


def noisy_oracle(data, eps1, seed=0, n=100):
    """ceiling: exact simulator with per-step isotropic state noise sized so the embedded one-step error RMS = eps1."""
    S = data.S; te = data.test_states[:n, 0]
    sd = data.train_states.reshape(-1, S["dim"]).std(0)
    rng = np.random.RandomState(seed)
    base = data.nrm(te)
    # calibrate kappa
    k = 1e-3
    for _ in range(30):
        e = np.sqrt(((data.nrm(te + k * sd * rng.randn(*te.shape)) - base) ** 2).mean())
        k *= eps1 / max(e, 1e-12)
    from selwm.p2_dyn import _rk4
    # noisy trajectory: perturb each step, compare with exact truth
    x = te.copy(); noisy = [data.nrm(x)]
    for _ in range(data.test.shape[1] - 1):
        x = _rk4(S["f"], x, S["dt"], S["sub"]) + k * sd * rng.randn(*x.shape)
        noisy.append(data.nrm(x))
    pr = torch.tensor(np.stack(noisy, 1)); tru = data.test[:n]
    err = ((pr - tru) ** 2).mean(-1).sqrt(); bad = err > 0.3
    first = torch.where(bad.any(1), bad.float().argmax(1), torch.full((len(tru),), tru.shape[1])).float()
    return float(first.mean()), float(k)


for seed in seeds:
    d = Data(name, seed=seed if role == "test" else 100 + seed, n_train=NTR, role=role)
    S = d.S; dt = S["dim"]
    X = d.train.reshape(-1, D_OBS)
    T = local_tangent_normal(X, dt, k=32)
    ae = train_ae(d, k=dt + 1, seed=seed)
    proj = lambda z: ae(z)
    res = {}

    def run(tag, model, proj_=None, extra=None):
        t0 = time.time(); r = evaluate(model, d, proj_)
        with torch.no_grad():
            te = d.test; e1 = float(((model(te[:, :-1]) - te[:, 1:]) ** 2).mean().sqrt())
        r.update(dict(system=name, phase=phase, seed=seed, method=tag, onestep=e1))
        if extra: r.update(extra)
        log(r); return r

    # teacher forcing and AE projection on top of it
    m_tf = fit(d, "tf", seed=seed, steps=STEPS, bs=BS, lr=LR); r_tf = run("tf", m_tf)
    run("tf+aeproj", m_tf, proj)
    if phase == "tune":
        for s in GRID:
            m = fit(d, "iso", seed=seed, steps=STEPS, bs=BS, lr=LR, sigma=s); run(f"iso@{s}", m, extra=dict(sigma=s))
            m = fit(d, "normal", seed=seed, steps=STEPS, bs=BS, lr=LR, sigma=s, tangents=T); run(f"normal@{s}", m, extra=dict(sigma=s))
        for p in (0.3, 0.6):
            m = fit(d, "ss", seed=seed, steps=STEPS, bs=BS, lr=LR, H=8, ss_p=p); run(f"ss@{p}", m, extra=dict(ss_p=p))
    else:
        cfg = json.load(open("/home/user/world_models/results/p2/%s_selected.json" % name))
        m = fit(d, "iso", seed=seed, steps=STEPS, bs=BS, lr=LR, sigma=cfg["iso"]); run("iso", m, extra=dict(sigma=cfg["iso"]))
        m_iso = m
        m = fit(d, "normal", seed=seed, steps=STEPS, bs=BS, lr=LR, sigma=cfg["normal"], tangents=T); run("normal", m, extra=dict(sigma=cfg["normal"]))
        m = fit(d, "ss", seed=seed, steps=STEPS, bs=BS, lr=LR, H=8, ss_p=cfg["ss_p"]); run("ss", m, extra=dict(ss_p=cfg["ss_p"]))
        run("iso+aeproj", m_iso, proj)
        m, hist = fit_scn(d, T, seed=seed, steps=STEPS, lr=LR, bs=BS); run("scn", m, extra=dict(sigma_hist=hist))
        v, k = noisy_oracle(d, r_tf["onestep"], seed)
        log(dict(system=name, phase=phase, seed=seed, method="noisy_oracle@tf_eps", vpt=v, kappa=k))
        v, k = noisy_oracle(d, r_tf["onestep"] / 2, seed)
        log(dict(system=name, phase=phase, seed=seed, method="noisy_oracle@tf_eps/2", vpt=v, kappa=k))
