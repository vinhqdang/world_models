"""Stage 0 diagnostic: how the learned residual-reuse R^2 depends on the predictor's error. The predictor is fitted on a
subsample of the 8192 episodes (sizes 512, 1024, 2048, 4096, 8192); the paper's own scaling table (Table 8) shows the
trained predictor reaching 22-38% with 819-6553 episodes through the frozen-encoder pipeline. Counterfactual R^2 of the
trained predictor is used here only to locate the paper-like regime, and no later stage uses this selection."""
import sys, json, glob
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import worlds, predictor as P, evalpop as EP, metrics as M
DATA = "/home/user/world_models/results/cwk/data"
OUT = "/home/user/world_models/results/cwk"
out = {}
for cell in (sys.argv[1:] or ["hand_affine100", "hand_affine50", "hand_l0"]):
    col = worlds.load(f"{DATA}/{cell}_col_8192x5.npz")
    pop = worlds.load(sorted(glob.glob(f"{DATA}/{cell}_pop_*.npz"))[0])
    truth = pop["truth"]
    N, E, K, D = truth.shape
    ev = M.Evaluator(truth, pop["weights"], [0, 1], n_seeds=5)
    for n_ep in (512, 1024, 2048, 4096):
        sub = {k: (v[:n_ep] if isinstance(v, np.ndarray) and v.ndim > 1 else v) for k, v in col.items()}
        H, a, y, s, ep, st = P.flatten_transitions(sub, t_max=3)
        tr, rr = [], []
        for seed in range(5):
            m, _ = P.fit_mean(H, a, y, s, ep, seed, epochs=40)
            mu = EP.mean_predictions(m, pop)
            tr.append(ev.loss(EP.broadcast(mu, E)))
            rr.append(ev.loss(EP.residual_reuse(mu, truth)))
        out[f"{cell}/{n_ep}"] = dict(trained=ev.r2(np.array(tr)), rr_learned=ev.r2(np.array(rr)), ideal=ev.r2(M.exact_mean_residual_loss(truth, [0, 1])))
        print(cell, n_ep, "trained %.1f  rr_learned %.1f  ideal %.1f" % tuple(100 * out[f"{cell}/{n_ep}"][k]["r2"] for k in ("trained", "rr_learned", "ideal")), flush=True)
json.dump(out, open(f"{OUT}/stage0_matched_{'_'.join(sys.argv[1:]) or 'all'}.json", "w"), indent=1)
