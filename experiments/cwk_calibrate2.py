"""Refinement of the E2/E3 calibration: a few (sigma, force) combinations near the step-1/2 estimates; the one whose
primary-group outcome-blind oracle R^2 is closest to 0.50 is stored (simulator only)."""
import sys, json
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds, metrics as M
from experiments.cwk_calibrate import save, DEV_OFF
key = sys.argv[1]
if key == "e2":
    fam, alim, combos = "cup", 0.35, [(0.58, 2.6), (0.60, 3.0), (0.62, 3.5)]
    mk = lambda sg, f: cells.e2_cell(0.5, sigma=sg, wind=f, seed_offset=DEV_OFF)
else:
    fam, alim, combos = "fetch", 0.48, [(0.66, 20000.0), (0.66, 28000.0), (0.66, 36000.0)]
    mk = lambda sg, f: cells.e3_cell(0.5, sigma=sg, load=f, seed_offset=DEV_OFF)
if len(sys.argv) > 2:
    combos = json.loads(sys.argv[2])
grp = cells.GROUPS[fam]
log = []
for sg, f in combos:
    pop = worlds.population(mk(sg, f), 48, 32)
    t = pop["truth"]
    row = dict(sigma=sg, force=f)
    for g, idx in grp["groups"].items():
        ev = M.Evaluator(t, pop["weights"], idx, n_seeds=1)
        row[g + "_oracle"] = ev.r2(M.oracle_loss(t, idx))["r2"]
        row[g + "_copy"] = ev.r2(ev.loss(np.repeat(t[:, :, :1], t.shape[2], 2)))["r2"]
        row[g + "_ideal"] = ev.r2(M.exact_mean_residual_loss(t, idx))["r2"]
        row[g + "_rho"] = M.interaction_ratio(t, idx)
    log.append(row)
    print(key, row, flush=True)
prim = grp["primary"]
best = min(log, key=lambda r: abs(r[prim + "_oracle"] - 0.5))
save(key, dict(sigma=best["sigma"], gain=0.98 * (1 - best["sigma"]) / alim, **({"wind": best["force"]} if key == "e2" else {"load": best["force"]}),
               primary=prim, refinement=log, note="two-step scan (cwk_calibrate.py) then refinement (cwk_calibrate2.py)"))
print("chosen", best)
