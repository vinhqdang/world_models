"""Simulator-only calibration of the paper cells (information-limited INFOLIM cells, MISALIGN) to the authors' design target:
outcome-blind oracle R^2 about 50% (+- 8) on the primary group. No model and no method comparison is involved. Development
cohorts use reset seeds disjoint from every evaluation population (cell seed offsets 500_000 / 600_000, MISALIGN 300_000 / 400_000).
The per-axis command budget sigma is searched (command gain g = 0.98 (1 - sigma) / action limit, so that the disturbed command
never leaves the native range); the force scale of r = d + 2 cells is force_unit * sigma, force_unit calibrated once per
environment (force-only driver to oracle R^2 0.70, as the Stage 2 two-stage calibration).
New keys are written to results/cwk/CALIBRATION.json (old keys untouched).
usage: python3 experiments/paper_calibrate.py ENV [ENV...]      ENV in E1o E2o E3o MIS"""
import sys, json, os, time, fcntl
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds, metrics as M

CAL = "/home/user/world_models/results/cwk/CALIBRATION.json"
TARGET, TOL = 0.5, 0.08
SIG_MIN, SIG_MAX = 0.25, 0.95


def save(key, val):
    with open(CAL + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cal = json.load(open(CAL)) if os.path.exists(CAL) else {}
        cal[key] = val
        json.dump(cal, open(CAL, "w"), indent=1)
        fcntl.flock(lk, fcntl.LOCK_UN)


def stats(world, fam, parents, draws):
    pop = worlds.population(world, parents, draws)
    g = cells.GROUPS[fam]
    idx = g["groups"][g["primary"]]
    t = pop["truth"]
    ev = M.Evaluator(t, pop["weights"], idx, n_seeds=1)
    K = t.shape[2]
    return dict(oracle=ev.r2(M.oracle_loss(t, idx))["r2"], copy=ev.r2(ev.loss(np.repeat(t[:, :, :1], K, 2)))["r2"],
                ideal=ev.r2(M.exact_mean_residual_loss(t, idx))["r2"], rho=M.interaction_ratio(t, idx))


def search(mk, fam, parents=40, draws=24, init=(0.6, 0.8), log=print):
    """mk(sigma) -> World on the development cohort. Monotone decreasing oracle(sigma): interpolate to the target."""
    pts = []

    def ev(sg):
        st = stats(mk(sg), fam, parents, draws)
        pts.append(dict(sigma=round(float(sg), 4), **st))
        log(f"    sigma {sg:.3f} oracle {st['oracle']:.3f} copy {st['copy']:.3f} ideal {st['ideal']:.3f} rho {st['rho']:.3f}")
        return st["oracle"]

    for sg in init:
        ev(sg)
    for _ in range(5):
        best = min(pts, key=lambda p: abs(p["oracle"] - TARGET))
        if abs(best["oracle"] - TARGET) < 0.02:
            break
        s = sorted(pts, key=lambda p: p["sigma"])
        lo = [p for p in s if p["oracle"] >= TARGET]
        hi = [p for p in s if p["oracle"] < TARGET]
        if lo and hi:      # bracket: interpolate between the closest points on each side
            a, b = lo[-1], hi[0]
        else:              # extrapolate with the two nearest points
            a, b = (s[0], s[1]) if not hi else (s[-2], s[-1])
        if abs(a["oracle"] - b["oracle"]) < 1e-6:
            break
        sg = a["sigma"] + (TARGET - a["oracle"]) * (b["sigma"] - a["sigma"]) / (b["oracle"] - a["oracle"])
        sg = float(np.clip(sg, SIG_MIN, SIG_MAX))
        if any(abs(sg - p["sigma"]) < 0.005 for p in pts):
            break
        ev(sg)
    best = min(pts, key=lambda p: abs(p["oracle"] - TARGET))
    return best, pts


def calibrate_info(env, name, law, pi, r, iid, log=print):
    fam = cells.INFO_ENV[env]["fam"]
    d = cells.info_dim(env)
    r = d + 1 if r is None else r
    key = cells.info_key(env, law, pi, r) + ("_iid50" if iid < 1 else "")
    alim = cells.INFO_ENV[env]["alim"]
    log(f"[{key}]")
    mk = lambda sg: cells.info_cell(env, law, pi, r, iid, sigma=sg, seed_offset=500_000)
    best, pts = search(mk, fam, init=(0.6, 0.8), log=log)
    # verification on a second, disjoint development cohort (48 parents x 32 draws)
    ver = stats(cells.info_cell(env, law, pi, r, iid, sigma=best["sigma"], seed_offset=600_000), fam, 48, 32)
    ok = abs(ver["oracle"] - TARGET) <= TOL
    log(f"    chosen sigma {best['sigma']} (dev oracle {best['oracle']:.3f}); verification cohort oracle {ver['oracle']:.3f} "
        f"copy {ver['copy']:.3f} rho {ver['rho']:.3f} -> {'within' if ok else 'OUTSIDE'} 50 +- 8")
    save(key, dict(sigma=best["sigma"], gain=0.98 * (1 - best["sigma"]) / alim, dev_oracle=best["oracle"], verify=ver,
                   in_window=bool(ok), capped=bool(best["sigma"] >= SIG_MAX - 1e-9 or best["sigma"] <= SIG_MIN + 1e-9),
                   env=env, law=law, pi=pi, r=r, iid=iid, scan=pts))
    return key, best, ver


def calibrate_force(env, sigma_ref, log=print):
    fam = cells.INFO_ENV[env]["fam"]
    d = cells.info_dim(env)
    grid = {"E1o": [0.3, 0.6, 1.2, 2.4, 4.8], "E2o": [1.0, 3.0, 9.0, 27.0], "E3o": [4000.0, 14000.0, 40000.0, 120000.0]}[env]
    log(f"[force_unit_{env}] sigma_ref {sigma_ref}")
    scan = []
    for F in grid:
        w = cells.info_cell(env, "gated", cells.PI_DEFAULT, d + 2, 1.0, sigma=sigma_ref, seed_offset=500_000)
        w.channels = [dict(c, scale=0.0) for c in w.channels]
        w.forces = [dict(w.forces[0], scale=F)]
        st = stats(w, fam, 40, 24)
        scan.append(dict(force=F, **st))
        log(f"    force {F:g} oracle {st['oracle']:.3f} rho {st['rho']:.3f}")
        if st["oracle"] < 0.55:
            break
    o = np.array([s["oracle"] for s in scan])
    fs = np.array([s["force"] for s in scan])
    if o.min() < 0.7 < o.max():
        o_, f_ = o[::-1], np.log(fs)[::-1]
        order = np.argsort(o_)
        F = float(np.exp(np.interp(0.7, o_[order], f_[order])))
    else:
        F = float(fs[np.argmin(abs(o - 0.7))])
    save(f"force_unit_{env}", dict(unit=F / sigma_ref, force=F, sigma_ref=sigma_ref, target_force_only_oracle=0.7, scan=scan))
    log(f"    force {F:g} -> unit {F / sigma_ref:g}")


def calibrate_mis(offset, log=print):
    env = "E2o"
    key = f"misalign_{env}_tanh" + ("" if offset else "_nooff")
    alim = cells.INFO_ENV[env]["alim"]
    log(f"[{key}]")
    mk = lambda sg: cells.misalign_cell(env, "tanh", offset, 1.0, sigma=sg, seed_offset=300_000)
    best, pts = search(mk, "cup", log=log)
    ver = stats(cells.misalign_cell(env, "tanh", offset, 1.0, sigma=best["sigma"], seed_offset=400_000), "cup", 48, 32)
    ok = abs(ver["oracle"] - TARGET) <= TOL
    log(f"    chosen sigma {best['sigma']}; verification oracle {ver['oracle']:.3f} -> {'within' if ok else 'OUTSIDE'} 50 +- 8")
    save(key, dict(sigma=best["sigma"], gain=0.98 * (1 - best["sigma"]) / alim, dev_oracle=best["oracle"], verify=ver,
                   in_window=bool(ok), capped=bool(best["sigma"] >= SIG_MAX - 1e-9 or best["sigma"] <= SIG_MIN + 1e-9), scan=pts))


def refine(key, log=print):
    """Second round for a key whose verification cohort fell outside 50 +- 8: re-search on the (disjoint) 48 x 32 cohort at offset
    600_000, verify on a third cohort of 64 x 32 at offset 700_000. The first-round record is kept under 'round1'."""
    v = json.load(open(CAL))[key]
    env, law, pi, r, iid = v["env"], v["law"], v["pi"], v["r"], v["iid"]
    fam, alim = cells.INFO_ENV[env]["fam"], cells.INFO_ENV[env]["alim"]
    log(f"[refine {key}] first round sigma {v['sigma']}, verification oracle {v['verify']['oracle']:.3f}")
    mk = lambda sg: cells.info_cell(env, law, pi, r, iid, sigma=sg, seed_offset=600_000)
    best, pts = search(mk, fam, parents=48, draws=32, init=(v["sigma"] - 0.08, v["sigma"] + 0.04), log=log)
    ver = stats(cells.info_cell(env, law, pi, r, iid, sigma=best["sigma"], seed_offset=700_000), fam, 64, 32)
    ok = abs(ver["oracle"] - TARGET) <= TOL
    log(f"    refined sigma {best['sigma']} (cohort-2 oracle {best['oracle']:.3f}); third-cohort oracle {ver['oracle']:.3f} -> {'within' if ok else 'OUTSIDE'} 50 +- 8")
    rec = dict(v)
    rec.pop("round1", None)
    save(key, dict(sigma=best["sigma"], gain=0.98 * (1 - best["sigma"]) / alim, dev_oracle=best["oracle"], verify=ver, in_window=bool(ok),
                   capped=bool(best["sigma"] >= SIG_MAX - 1e-9 or best["sigma"] <= SIG_MIN + 1e-9), env=env, law=law, pi=pi, r=r, iid=iid,
                   scan=pts, round1=rec, refined=True))


if __name__ == "__main__":
    if sys.argv[1] == "refine":
        for k in sys.argv[2:]:
            refine(k)
        sys.exit(0)
    for env in sys.argv[1:]:
        t0 = time.time()
        if env == "MIS":
            calibrate_mis(True)
            calibrate_mis(False)
            continue
        d = cells.info_dim(env)
        kP, bP, _ = calibrate_info(env, "P", "gated", 0.2, None, 1.0)
        calibrate_force(env, bP["sigma"])
        for law, pi, r, iid in [("tanh", None, None, 1.0), ("gauss", None, None, 1.0), ("gated", 0.05, None, 1.0),
                                ("gated", 0.1, None, 1.0), ("gated", 0.5, None, 1.0), ("gated", 1.0, None, 1.0),
                                ("gated", 0.2, d, 1.0), ("gated", 0.2, d + 2, 1.0), ("gated", 0.2, None, 0.5)]:
            calibrate_info(env, "", law, pi if pi is not None else 0.2, r, iid)
        print(env, "done", round(time.time() - t0), "s", flush=True)
