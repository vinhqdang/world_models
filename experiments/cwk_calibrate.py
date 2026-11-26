"""Simulator-only calibration of the Stage 2 cells to the authors' design target (outcome-blind oracle R^2 about 50% on
the primary group, factual copy below 60%). No model and no method comparison is involved. Development cohorts use
reset seeds disjoint from every evaluation population.
usage: python3 experiments/cwk_calibrate.py KEY   (KEY in e1_nu50_gain, e1_nu80_gain, e1_nu50_sat, e1_nu50_deadzone, e2, e3)"""
import sys, json, os, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds, metrics as M

CAL = "/home/user/world_models/results/cwk/CALIBRATION.json"
DEV_OFF = 9_900_000


def stats(world, fam, parents=48, draws=32):
    pop = worlds.population(world, parents, draws)
    g = cells.GROUPS[fam]
    idx = g["groups"][g["primary"]]
    t = pop["truth"]
    ev = M.Evaluator(t, pop["weights"], idx, n_seeds=1)
    K = t.shape[2]
    return dict(oracle=ev.r2(M.oracle_loss(t, idx))["r2"], copy=ev.r2(ev.loss(np.repeat(t[:, :, :1], K, 2)))["r2"],
                ideal=ev.r2(M.exact_mean_residual_loss(t, idx))["r2"], rho=M.interaction_ratio(t, idx))


def save(key, val):
    cal = json.load(open(CAL)) if os.path.exists(CAL) else {}
    cal[key] = val
    json.dump(cal, open(CAL, "w"), indent=1)


if __name__ == "__main__":
    key = sys.argv[1]
    log = []
    if key.startswith("e1"):
        _, nu, prof = key.split("_")
        nu, fam = int(nu[2:]) / 100, "hand"
        for sigma in [0.45, 0.55, 0.65, 0.75, 0.85]:
            w = cells.e1_cell(nu, prof, 0.5, sigma=sigma, seed_offset=DEV_OFF)
            st = stats(w, fam)
            log.append(dict(sigma=sigma, **st))
            print(key, log[-1], flush=True)
        best = min(log, key=lambda r: abs(r["oracle"] - 0.5))
        save(key, dict(sigma=best["sigma"], gain=0.98 * (1 - best["sigma"]) / 0.45, scan=log))
    elif key in ("e2", "e3"):
        fam = "cup" if key == "e2" else "fetch"
        alim = 0.35 if key == "e2" else 0.48
        mk = (lambda sg, f: cells.e2_cell(0.5, sigma=sg, wind=f, seed_offset=DEV_OFF)) if key == "e2" else \
             (lambda sg, f: cells.e3_cell(0.5, sigma=sg, load=f, seed_offset=DEV_OFF))
        grp = cells.GROUPS[fam]
        prim = grp["primary"]
        def stats_g(world):
            pop = worlds.population(world, 40, 32)  # primary group listed first
            t = pop["truth"]
            out = {}
            for g, idx in grp["groups"].items():
                ev = M.Evaluator(t, pop["weights"], idx, n_seeds=1)
                out[g] = dict(oracle=ev.r2(M.oracle_loss(t, idx))["r2"], rho=M.interaction_ratio(t, idx))
            return out
        # step 1: gain driver alone (no force) -> oracle about 0.70 on the primary group
        scan1 = []
        for sg in (0.4, 0.55, 0.7, 0.85):
            st = stats_g(mk(sg, 0.0)); scan1.append(dict(stage=1, sigma=sg, **{f"{g}_oracle": v["oracle"] for g, v in st.items()}))
            print(key, scan1[-1], flush=True)
        o = np.array([r[f"{prim}_oracle"] for r in scan1]); sgs = np.array([r["sigma"] for r in scan1])
        sg = float(np.interp(0.70, o[::-1], sgs[::-1])) if o.min() < 0.7 < o.max() else float(sgs[np.argmin(abs(o - 0.7))])
        # step 2: force alone (gain channel amplitude ~0) with the same command gain -> oracle about 0.70
        def force_only(f):
            w = mk(sg, f)
            w.channels = [dict(c, scale=0.0) for c in w.channels]
            return stats_g(w)
        base_force = 0.25 if key == "e2" else 3000.0
        scan2 = []
        for f in [base_force * (4 ** i) for i in range(5)]:
            st = force_only(f); scan2.append(dict(stage=2, force=f, **{f"{g}_oracle": v["oracle"] for g, v in st.items()}))
            print(key, scan2[-1], flush=True)
        o2 = np.array([r[f"{prim}_oracle"] for r in scan2]); fs = np.array([r["force"] for r in scan2])
        f_star = float(np.exp(np.interp(0.70, o2[::-1], np.log(fs)[::-1]))) if o2.min() < 0.7 < o2.max() else float(fs[np.argmin(abs(o2 - 0.7))])
        st = stats_g(mk(sg, f_star))
        comb = dict(stage=3, sigma=sg, force=f_star, **{f"{g}_oracle": v["oracle"] for g, v in st.items()}, **{f"{g}_rho": v["rho"] for g, v in st.items()})
        print(key, comb, flush=True)
        best = comb
        save(key, dict(sigma=sg, gain=0.98 * (1 - sg) / alim, **({"wind": f_star} if key == "e2" else {"load": f_star}),
                       primary=prim, scan1=scan1, scan2=scan2, combined=comb))
    print("chosen", best)
