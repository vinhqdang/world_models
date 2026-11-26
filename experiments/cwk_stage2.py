"""Stage 2 gate P-A: rank >= 2 cells. usage: python3 experiments/cwk_stage2.py run CELL... | report"""
import sys, json, glob, os, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds, runner as RN, metrics as M

DATA = "/home/user/world_models/results/cwk/data"
OUT = "/home/user/world_models/results/cwk"
BASE_ARMS = ["pck_pol", "pck_rank1", "ls_signed", "ls_signed_nug", "ls_unsigned", "flow", "indist_gain", "indist_plain", "indist_struct",
             "hh_binned", "hh_global", "cvae", "rho_ref"]
CFG = dict(epochs=20, batch=512, steps=3000, width=384, n_seeds=5, threads=2, bins=8, cvae_steps=4000)
SEEDS = range(5)
MAIN = {"E1": ["E1_nu50_gain_iid50", "E1_nu80_gain_iid50"], "E2": ["E2_cup_iid50"], "E3": ["E3_fetch_iid50"]}
MISSPEC = ["E1_nu50_sat_iid50", "E1_nu50_deadzone_iid50"]
ORDER = ["E1_nu50_gain_iid50", "E1_nu80_gain_iid50", "E2_cup_iid50", "E3_fetch_iid50", "E1_nu50_sat_iid50",
         "E1_nu50_deadzone_iid50", "E1_nu50_gain_iid100", "E1_nu80_gain_iid100", "E2_cup_iid100", "E3_fetch_iid100",
         "E1_nu50_sat_iid100", "E1_nu50_deadzone_iid100"]


def arms_for(world):
    return BASE_ARMS + (["pck_echo", "pck_joint"] if world.iid_fraction < 1 else [])


def load_cell(name):
    w = cells.STAGE2[name]()
    col = worlds.load(glob.glob(f"{DATA}/{name}_col_*.npz")[0])
    pop = worlds.load(glob.glob(f"{DATA}/{name}_pop_512x64.npz")[0])
    dev = worlds.load(glob.glob(f"{DATA}/{name}_dev_*.npz")[0])
    return w, cells.pipeline_cell(w, col, pop, dev)


def run(names):
    for name in names:
        t = time.time()
        w, cell = load_cell(name)
        RN.run_seeds(cell, arms_for(w), SEEDS, CFG, "s2")
        print(name, "done", round(time.time() - t), "s", flush=True)


def result(name):
    raws = [json.load(open(p)) for p in sorted(glob.glob(f"{RN.RAW}/s2_{name}_s*.json"))]
    if len(raws) < 5:
        return None, None
    w, cell = load_cell(name)
    return cell, RN.CellResult(cell, raws, RN.deterministic(cell), 5)


def fmt(r):
    return f"{100 * r['r2']:.1f} [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]"


def report():
    cal = json.load(open(f"{OUT}/CALIBRATION.json"))
    protocol = f"""
## Protocol as implemented (deviations in bold)

- Cells (built from the authors' simulators; physics untouched, mechanisms composed by `cwk/sim.py`). Amplitudes were calibrated on simulator-only
  development cohorts (disjoint reset seeds, 40-48 parents x 32 draws; no model involved) to the authors' target of an outcome-blind oracle R^2 of
  about 50% on the scored group: E1 = HandReach, independent offset driver (weight 1 - nu) and gain driver (weight nu), total amplitude sigma, command gain
  g = 0.98 (1 - sigma) / 0.45: nu = 0.5 (sigma {cal['e1_nu50_gain']['sigma']}, g {cal['e1_nu50_gain']['gain']}), nu = 0.8 (sigma {cal['e1_nu80_gain']['sigma']}, g {cal['e1_nu80_gain']['gain']});
  misspecified variants nu = 0.5 with a saturating gain tanh(3x)/tanh(3) (sigma {cal['e1_nu50_sat']['sigma']}) and a deadzone gain (zero for |x| < 0.4, sigma {cal['e1_nu50_deadzone']['sigma']}).
  E2 = Ball-in-Cup, gain driver (sigma {cal['e2']['sigma']}, g {round(cal['e2']['gain'], 3)}) plus an independent wind force on the ball (scale {cal['e2']['wind']}); E3 = FetchReach, gain driver on the goal command
  (sigma {cal['e3']['sigma']}, g {round(cal['e3']['gain'], 3)}) plus an independent load on the gripper link (scale {cal['e3']['load']} N).
  **Deviation 1 (E1 parametrization):** a pure-gain driver (lambda = 1) is rank one and total amplitude sigma(1 - lambda + lambda) <= 1 - g a_lim
  cannot accommodate two full-size drivers, so E1 is indexed by the gain weight nu in {{0.5, 0.8}} (interaction ratio rho 0.51 and 1.71, matching the authors' lambda = 0.5 and
  close to lambda = 1) instead of lambda in {{0.5, 1}}.
- Drivers: eps_j = tanh(driver_j), driver_j = sqrt(1 - v) field_j(t) + sqrt(v) n_t (authors' EpisodeField, temporal scale 4); v = 0.5 main cells (true lag-1 correlation
  of eps 0.464, lag-2 0.423), v = 1 for the iid cells. Episodes of 5 transitions (t = 0, 1 behaviour policy; t = 2 anchor, 3, 4 uniform actions), 16384 factual
  episodes per cell; evaluation population 512 parents x 64 draws x 3 queries (authors' protocol, state level); 64 x 32 development population (separate seeds) used only
  for the rho-interpolation reference arm.
- Mean predictor mu_hat(H, a): MLP (3 x 384, SiLU) on the last three states and the action, 2-fold cross-fitting by episode for the residuals, a model on all data for the queries.
  **Deviation 2 (compute):** 20 epochs, batch 512 (Stage 0 used 40 epochs, batch 256, width 512); kernel heads 3000 steps of batch 512; CVAE 4000 steps; 5 fit seeds throughout.
- **Deviation 3 (coordinates):** every arm models the residual of the scored coordinates only (wrist 2, cup position 2, gripper position 3), not all joint angles; a first
  version modelling all 24 joint angles under-fitted the neural head (see `cwk/cells.py`), and the choice was made on held-out factual NLL and compute (and seed-0 numbers were visible).
  FetchReach finger coordinates are fixed by the declared fixed action and are not scored.
- Kernel arms (psi(a) = (1, a_axes / a_lim) with the declared actuated axes; this is the same structural knowledge the abduction baselines use): `pck_pol` factor model rank 4, quasi-likelihood of single residuals with a learned diagonal
  nugget, transport with N = 0 and ridge tau = 1e-3 (standardised units); `pck_echo` unstructured Phi(h, a) of rank 8 fitted to consecutive-residual pairs (diagonal blocks, lag-1 products
  with a learned kappa); `pck_joint` the psi-structured factor model fitted to the same pair likelihood; `pck_rank1` rank-one polarization head (vector signed location-scale);
  `ls_signed` / `ls_signed_nug` / `ls_unsigned` per-coordinate affine signed (or absolute) scale; `flow` heteroscedastic Cholesky (full-covariance) abduction; `indist_gain` single-driver
  input-disturbance abduction with the nominal fault type (offset share, gain share, amplitude of the cell) and `indist_plain` offset-only; `indist_struct`* the exact command channels;
  `hh_binned` binned (8 k-means bins of the context) Hildreth-Houck moment estimator with Burer-Monteiro factors and the same kriging, `hh_global` one bin (the context-free closed form);
  `cvae` CF-GPS with a learned conditional VAE (z in R^4); `rho_ref`* interpolation tuned on the development population's counterfactuals; `oracle`*, `rr_exact`*, `oracle_kernel`* (Monte-Carlo
  kriging with the parent's own conditional moments, two folds over draws).
- Best baseline = highest pooled R^2 among non-starred arms that are not the merged algorithm's own variants (`pck_pol`, `pck_echo`, `pck_joint`, `hh_global`); `pck_rank1`, `hh_binned`, `cvae`
  are therefore baselines. P-A aggregation: an environment passes if, in every one of its main (iid 0.5, well-specified) cells, joint - best baseline >= 0.10 with the paired CI above 0;
  Holm over the 3 environments uses, per environment, the largest cell p-value (two-sided paired bootstrap); the gate needs >= 2 environments and the misspecified condition
  (joint >= rr_learned - 0.02 on the saturating and deadzone cells).
"""
    md = ["# Stage 2: gate P-A (rank >= 2 disturbances)\n", protocol,
          "Physical counterfactual R^2 (%) with whole-parent x fit-seed bootstrap 95% CI (5 fit seeds). "
          "Arms marked * use privileged simulator information (exact means, exact structure, or counterfactual tuning) "
          "and are excluded from the best-baseline comparison. Differences are paired.\n"]
    summary = {}
    for name in ORDER:
        cell, res = result(name)
        if res is None:
            md.append(f"\n_{name}: not complete_\n")
            continue
        g = cell.primary
        md.append(f"\n## {name} (primary group {g}, rho = {res.rho[g]:.2f})\n")
        for grp in cell.groups:
            tb = res.table(grp)
            md.append(f"\n**{grp}**\n\n| arm | R^2 % [95% CI] |\n|---|---|")
            for a in ["oracle", "copy", "rr_exact", "oracle_kernel", "rho_ref", "indist_struct", "trained", "rr_learned"] + \
                    [a for a in res.arms if a not in ("trained", "rr_learned", "rho_ref", "indist_struct")]:
                if a in tb:
                    star = "*" if a in RN.PRIVILEGED else ""
                    md.append(f"| {a}{star} | {fmt(tb[a])} |")
        best, scores = res.best_baseline(g)
        row = dict(best_baseline=best, best_r2=scores[best])
        for m in ("pck_joint", "pck_pol", "pck_echo"):
            if m in res.L[g]:
                dd = res.diff(g, m, best)
                row[m] = dict(r2=res.r2(g, m), vs_best=dd, vs_rr_learned=res.diff(g, m, "rr_learned"),
                              vs_rr_exact=res.diff(g, m, "rr_exact"))
                md.append(f"\n{m} minus best baseline ({best}): {100 * dd['diff']:+.1f} points, paired CI "
                          f"[{100 * dd['ci95'][0]:+.1f}, {100 * dd['ci95'][1]:+.1f}], p={dd['p_boot']:.3f}")
                d2 = row[m]["vs_rr_learned"]
                md.append(f"\n{m} minus rr_learned: {100 * d2['diff']:+.1f} [{100 * d2['ci95'][0]:+.1f}, {100 * d2['ci95'][1]:+.1f}]")
        best2, sc2 = res.best_baseline(g, exclude=("hh_binned",))
        for m in ("pck_joint", "pck_pol"):
            if m in res.L[g]:
                dd2 = res.diff(g, m, best2)
                row[m]["vs_best_excluding_hh_binned"] = dict(baseline=best2, **dd2)
                md.append(f"\nSensitivity (binned Hildreth-Houck BLUP removed from the baseline set): {m} minus best remaining baseline "
                          f"({best2}, {100 * sc2[best2]:.1f}%): {100 * dd2['diff']:+.1f} points, paired CI [{100 * dd2['ci95'][0]:+.1f}, {100 * dd2['ci95'][1]:+.1f}]")
        nll = {k: float(np.mean([d[k] for d in res.diag if k in d])) for k in res.diag[0] if k.startswith(("nll_", "kappa_"))}
        row["factual_heldout_nll_and_kappa"] = nll
        ov = [d["overid"] for d in res.diag if "overid" in d]
        if ov:
            row["overid"] = ov
            md.append("\nFactual held-out NLL / kappa: " + json.dumps({k: round(v, 3) for k, v in nll.items()}))
            md.append("\nOveridentification (polarization vs echo cross-kernel), per fit seed: " + json.dumps(
                [{k: round(v, 4) if isinstance(v, float) else v for k, v in o.items()} for o in ov]))
        summary[name] = row
    # ---------------- summary table ---------------------------------------------------------------------------------
    head = ["\n## Summary (primary group; R^2 %, 5 fit seeds)\n",
            "| cell | iid fraction | rho | rr_learned | best baseline (R^2) | oracle kernel* | pck_pol | pck_echo | pck_joint | joint - best baseline [95% CI] | joint - pol [95% CI] | joint - best excl. binned HH [95% CI] | kappa_hat (joint; true 0.464) | overid z (mean of 5 seeds) | overid rel. discrepancy |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        cell, res = result(name)
        if res is None:
            continue
        g = cell.primary
        best, sc = res.best_baseline(g)
        b2, sc2 = res.best_baseline(g, exclude=("hh_binned",))
        has = "pck_joint" in res.L[g]
        def cell_r(a):
            return f"{100 * res.r2(g, a)['r2']:.1f}" if a in res.L[g] else "n/a"
        def dstr(m, o):
            if m not in res.L[g]:
                return "n/a"
            d = res.diff(g, m, o)
            return f"{100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
        m = "pck_joint" if has else "pck_pol"
        head.append(f"| {name} | {cell.iid} | {res.rho[g]:.2f} | {cell_r('rr_learned')} | {best} ({100 * sc[best]:.1f}) | {cell_r('oracle_kernel')} | "
                    f"{cell_r('pck_pol')} | {cell_r('pck_echo')} | {cell_r('pck_joint')} | {dstr(m, best)}{'' if has else ' (pol)'} | {dstr('pck_joint', 'pck_pol')} | {dstr(m, b2)} (vs {b2}) | "
                    + (f"{np.mean([d['kappa_joint'] for d in res.diag]):.2f} | {np.mean([d['overid']['z'] for d in res.diag]):.1f} | {np.mean([d['overid']['rel_discrepancy'] for d in res.diag]):.2f} |" if has else "n/a | n/a | n/a |"))
    md[3:3] = head
    md.append("""
## Reading of the results

- P-A is decided by the baseline set. The binned Hildreth-Houck BLUP (moment estimator of the same bi-affine kernel class per context bin, no network, no amortization) is within a few points of the
  neural heads in every cell (joint minus it: -2.4, -2.0, -0.7, +0.3 points on the four main cells), so no cell comes near the +10 margin. With that baseline removed, the joint estimator leads the best remaining
  baseline by +9.3 (E1 nu 0.5, CF-GPS CVAE), +19.8 (E1 nu 0.8), +15.7 (E2) and +16.7 (E3) points; the cells' rank-two structure is what the rank-one signed rules, flows, input-disturbance
  abduction and the CVAE cannot represent, but a classical estimator represents it too.
- Persistence did not help: joint is 0.3-1.2 points below polarization-only in the main cells, echo-only is 0.4-7.6 points below it, and the estimated persistence kappa_hat (0.2-0.3) is about half of the true
  lag-1 correlation of the driver (0.464). A plausible cause (not tested here) is that the mean predictor sees the previous states, which reveal the previous disturbance, and so absorbs part of the persistent
  component before the residuals are formed; lagged residual products then identify an attenuated kappa. The overidentification statistic between the polarization-implied and echo-implied cross-kernels (z of the held-out lag-product loss difference,
  and the relative discrepancy of the two cross-kernels) is large in five of the six iid 0.5 cells (|z| 4.0-7.1; relative discrepancy 0.11-0.28 where z is positive; E1 nu 0.8 is quiet at z 1.1) although the polarization-implied kernel is the accurate one
  (it ties the binned moment estimator and sits within 1-6 points of the oracle kernel): as implemented the statistic reacts to the estimation problem of the echo route (attenuated kappa_hat, an unstructured network), not to
  misspecification of the bi-affine class, and is not calibrated (Stage 3 was not attempted).
- Misspecified psi (saturating gain, deadzone): the joint estimator stays far above residual reuse (+28.8 and +11.2 points) and ties the binned moment estimator, but loses 16 and 8 points to the exact-structure oracle kernel.
- iid_fraction = 1.0 cells (polarization only): polarization-only minus the best baseline (binned HH) is +0.8 (E2), +0.3 (E3) and -1.2 (E1 nu 0.5), i.e. again a tie with the classical estimator, and +19 to +23 points over the best baseline that is not a moment estimator of the same class.
""")
    # ---------------- verdict -------------------------------------------------------------------------------------
    md.append("\n# P-A verdict\n")
    verdict = {}
    env_p, env_ok = {}, {}
    for env, names in MAIN.items():
        ok, ps, details = True, [], []
        for n in names:
            r = summary.get(n)
            if r is None or "pck_joint" not in r:
                ok = False
                details.append(f"{n}: missing")
                continue
            dd = r["pck_joint"]["vs_best"]
            ok &= (dd["diff"] >= 0.10) and (dd["ci95"][0] > 0)
            ps.append(dd["p_boot"])
            details.append(f"{n}: joint - {r['best_baseline']} = {100 * dd['diff']:+.1f} pts, CI [{100 * dd['ci95'][0]:+.1f}, {100 * dd['ci95'][1]:+.1f}]")
        env_p[env] = max(ps) if ps else 1.0
        env_ok[env] = (ok, details)
    envs = list(MAIN)
    rej, adj = M.holm([env_p[e] for e in envs])
    passed = [e for e, rj in zip(envs, rej) if rj and env_ok[e][0]]
    for e, a, rj in zip(envs, adj, rej):
        md.append(f"- {e}: effect and CI criterion {'met' if env_ok[e][0] else 'NOT met'}; Holm-adjusted p = {a:.4f} "
                  f"({'reject' if rj else 'not significant'}); " + "; ".join(env_ok[e][1]))
    mis_ok = True
    for n in MISSPEC:
        r = summary.get(n)
        if r is None or "pck_joint" not in r:
            mis_ok = False
            md.append(f"- misspecified {n}: missing")
            continue
        dd = r["pck_joint"]["vs_rr_learned"]
        good = dd["diff"] >= -0.02
        mis_ok &= good
        md.append(f"- misspecified {n}: joint - rr_learned = {100 * dd['diff']:+.1f} pts [{100 * dd['ci95'][0]:+.1f}, {100 * dd['ci95'][1]:+.1f}] -> {'ok' if good else 'FAIL'}"
                  f"; joint - rr_exact = {100 * r['pck_joint']['vs_rr_exact']['diff']:+.1f} pts")
    P_A = (len(passed) >= 2) and mis_ok
    md.append(f"\n**Environments passing (strict: all main cells, Holm alpha 0.05): {passed} ({len(passed)}/3). "
              f"Misspecified-psi condition: {'met' if mis_ok else 'NOT met'}. GATE P-A: {'PASS' if P_A else 'FAIL'}.**")
    # sensitivity: the same rule with the binned Hildreth-Houck BLUP removed from the baseline set (not licensed by the protocol)
    env_p2, env_ok2 = {}, {}
    for env, names in MAIN.items():
        ok, ps = True, []
        for n in names:
            r = summary.get(n)
            if r is None or "pck_joint" not in r:
                ok = False
                continue
            dd = r["pck_joint"]["vs_best_excluding_hh_binned"]
            ok &= (dd["diff"] >= 0.10) and (dd["ci95"][0] > 0)
            ps.append(dd["p_boot"])
        env_p2[env], env_ok2[env] = (max(ps) if ps else 1.0), ok
    rej2, adj2 = M.holm([env_p2[e] for e in envs])
    passed2 = [e for e, rj in zip(envs, rej2) if rj and env_ok2[e]]
    md.append(f"\nSensitivity (NOT the protocol's verdict): with the binned Hildreth-Houck BLUP removed from the baseline set the same rule gives environments passing = {passed2} "
              f"(E1 fails through its nu = 0.5 cell, +9.3 [+8.1, +10.5]); the gate would then be {'PASS' if (len(passed2) >= 2 and mis_ok) else 'FAIL'}. "
              "The protocol lists the binned Hildreth-Houck BLUP as a baseline, so the official verdict above stands.")
    verdict = dict(passed=passed, misspec_ok=mis_ok, P_A=P_A, holm_adjusted=dict(zip(envs, adj)), sensitivity_without_hh_binned=dict(passed=passed2))
    open(f"{OUT}/STAGE2.md", "w").write("\n".join(md))
    json.dump(dict(summary=summary, verdict=verdict), open(f"{OUT}/stage2.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:] or ORDER)
    else:
        report()
