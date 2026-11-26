"""Unit tests of the kernel heads on analytic worlds with a known conditional kernel.
usage: python3 experiments/cwk_test_kernels.py [quick]"""
import sys, time, json
sys.path.insert(0, "/home/user/world_models")
import numpy as np, torch
from cwk import analytic, kernels as KR, predictor as P, pipeline as PL, metrics as M, evalpop as EP, oracle as OR

def cell_for(world, col, pop, dev=None, nominal=None):
    d = world.d
    return PL.Cell("analytic", col, pop, coords=list(range(d)), groups={"all": list(range(d))}, axes=world.axes, a_lim=1.0,
                   act_gain=world.gain, nominal=nominal or [dict(kind="offset", scale=0.1)],
                   true_cmd=world.channels, dev=dev, iid=world.iid)

def kernel_recovery(world, head, kd, n=3000, seed=0):
    """Relative Frobenius error of G G'^T against the analytic K (iid worlds), in standardised coordinates."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(kd.n, n, replace=False)
    h, psi = kd.h[idx], kd.psi[idx]
    a2 = rng.uniform(-1, 1, (n, world.d))
    psi2 = torch.as_tensor(KR.psi_features(a2, world.axes, 1.0)[0], dtype=torch.float32)
    a1 = kd.afeat[idx].numpy()
    # context c: last coordinate of the current state is not in kd directly -> use stored raw contexts
    return idx, a1, a2

if __name__ == "__main__":
    quick = len(sys.argv) > 1
    n_col, steps = (3000, 1500) if quick else (12000, 4000)
    out = {}
    for name, w in [("dial_l1_iid05", analytic.dial(1.0, d=2, iid=0.5, seed=1)),
                    ("two_driver_iid05", analytic.two_driver(0.6, d=2, iid=0.5, seed=2)),
                    ("two_driver_iid1", analytic.two_driver(0.6, d=2, iid=1.0, seed=3))]:
        t = time.time()
        col = w.collect(n_col, T=5)
        pop = w.population(128, 64)
        dev = w.population(64, 32, start=10_000)
        nominal = [dict(kind="affine", g=0.6, scale=sum(c["scale"] for c in w.channels))]
        cell = cell_for(w, col, pop, dev, nominal)
        arms = ["pck_pol", "ls_signed", "ls_unsigned", "flow", "hh_global", "hh_binned", "indist_gain", "indist_struct",
                "rho_ref", "cvae"] + (["pck_echo", "pck_joint"] if w.iid < 1 else [])
        res = PL.run_seed(cell, 0, arms, dict(epochs=15, steps=steps, width=256, cvae_steps=2500, n_seeds=1), log=lambda *a, **k: None)
        truth = pop["truth"]
        ev = M.Evaluator(truth, pop["weights"], [0, 1], n_seeds=1)
        table = {k: round(ev.r2(v["all"])["r2"], 3) for k, v in res["losses"].items()}
        table["oracle(no outcome)"] = round(ev.r2(M.oracle_loss(truth, [0, 1]))["r2"], 3)
        table["rr_exact"] = round(ev.r2(M.exact_mean_residual_loss(truth, [0, 1]))["r2"], 3)
        mc = OR.mc_kriging(truth, [0, 1])
        table["oracle_kernel(MC)"] = round(ev.r2(ev.loss(np.concatenate([mc, np.zeros(mc.shape[:3] + (1,))], -1))) ["r2"], 3)
        print(name, "rho", round(M.interaction_ratio(truth, [0, 1]), 3), json.dumps(table), {k: v for k, v in res["diag"].items() if k != "overid"},
              res["diag"].get("overid"), f"{time.time() - t:.0f}s", flush=True)
        out[name] = table
    json.dump(out, open("/home/user/world_models/results/cwk/unit_kernels.json", "w"), indent=1)
