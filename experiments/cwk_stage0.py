"""Stage 0: replication of the authors' state-level HandReach results (lambda = 0, 0.5, 1).
usage: python3 experiments/cwk_stage0.py [cells...]   (needs results/cwk/data/*)"""
import sys, json, time, os, glob
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import worlds, predictor as P, evalpop as EP, metrics as M

DATA = "/home/user/world_models/results/cwk/data"
OUT = "/home/user/world_models/results/cwk"
GROUPS = {"wrist": [0, 1], "joint_angles": list(range(24))}
PAPER = {  # Table 11 (MAE-B / DINOv2-S rows), Table 4 screen: percent R^2 on the wrist
    "hand_l0": dict(oracle=49.1, trained=(38.1, 39.9), cf_feature=(82.4, 82.3), cf_readout=(85.6, 86.1), ideal=98.3),
    "hand_affine50": dict(oracle=52.9, trained=(41.0, 43.3), cf_feature=(70.5, 70.6), cf_readout=(67.4, 66.3), ideal=78.9),
    "hand_affine100": dict(oracle=52.4, trained=(41.4, 41.6), cf_feature=(6.4, 11.0), cf_readout=(-11.8, -12.8), ideal=-1.1),
}
DEFAULT_SIZE = {"hand_l0": (128, 128), "hand_affine50": (128, 64), "hand_affine100": (128, 64)}


def subset(pop, parents, draws):
    d = dict(pop)
    for k in ("sources", "past_actions", "past_eps", "actions", "posterior"):
        d[k] = pop[k][:parents]
    d["eps"] = pop["eps"][:parents, :draws]
    d["truth"] = pop["truth"][:parents, :draws]
    rng = np.random.default_rng(1)
    d["weights"] = rng.multinomial(parents, np.full(parents, 1 / parents), size=512)
    return d


def evaluate(pop, models, tag):
    truth = pop["truth"]
    N, E, K, D = truth.shape
    mus = [EP.mean_predictions(m, pop) for m in models]
    res = {}
    for g, coords in GROUPS.items():
        ev = M.Evaluator(truth, pop["weights"], coords, n_seeds=len(models))
        trained = np.stack([ev.loss(EP.broadcast(mu, E)) for mu in mus])
        rr = np.stack([ev.loss(EP.residual_reuse(mu, truth)) for mu in mus])
        res[g] = dict(
            oracle=ev.r2(M.oracle_loss(truth, coords)),
            copy=ev.r2(ev.loss(np.repeat(truth[:, :, :1], K, 2))),
            ideal=ev.r2(M.exact_mean_residual_loss(truth, coords)),
            trained=ev.r2(trained), rr_learned=ev.r2(rr),
            rho=M.interaction_ratio(truth, coords),
            rr_minus_ideal=ev.diff(rr, M.exact_mean_residual_loss(truth, coords)),
        )
    return res


def main(cells):
    out = {}
    for cell in cells:
        pops = glob.glob(f"{DATA}/{cell}_pop_*.npz")
        col = worlds.load(glob.glob(f"{DATA}/{cell}_col_8192x5.npz")[0])
        pop = worlds.load(sorted(pops)[0])
        H, a, y, s, ep, st = P.flatten_transitions(col, t_max=3)
        models = []
        t = time.time()
        for seed in range(5):
            m, vl = P.fit_mean(H, a, y, s, ep, seed, epochs=40)
            models.append(m)
            print(cell, "seed", seed, "val", round(vl, 4), round(time.time() - t), "s", flush=True)
        n, e = DEFAULT_SIZE[cell]
        out[cell] = {
            "default": dict(parents=n, draws=e, **evaluate(subset(pop, n, e), models, "default")),
            "large": dict(parents=pop["truth"].shape[0], draws=pop["truth"].shape[1], **evaluate(pop, models, "large")),
            "paper": PAPER[cell],
        }
        json.dump(out[cell], open(f"{OUT}/stage0_{cell}.json", "w"), indent=1)
        print(json.dumps({k: {g: round(v['r2'] * 100, 1) if isinstance(v, dict) and 'r2' in v else None
                              for g, v in out[cell]['large']['wrist'].items()} for k in ['large']}), flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or ["hand_l0", "hand_affine50", "hand_affine100"])
