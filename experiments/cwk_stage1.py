"""Stage 1 honesty check: polarization-only PCK against signed/unsigned location-scale, residual reuse, flow abduction
and gain-rescaled input-disturbance abduction on the analytic dial and the HandReach dial cells (all rank one).
usage: python3 experiments/cwk_stage1.py run analytic|hand [cells...]   |   report"""
import sys, json, glob, os, time
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import analytic, cells, worlds, pipeline as PL, runner as RN, metrics as M

DATA = "/home/user/world_models/results/cwk/data"
OUT = "/home/user/world_models/results/cwk"
ARMS = ["pck_pol", "pck_rank1", "ls_signed", "ls_signed_nug", "ls_unsigned", "flow", "indist_gain", "indist_plain", "hh_global"]
CFG_AN = dict(epochs=20, steps=3000, width=256, n_seeds=5, threads=2)
CFG_HAND = dict(epochs=25, batch=512, steps=3000, width=384, n_seeds=5, threads=2)
SEEDS = range(5)

ANALYTIC = {f"dial_d{d}_l{int(l * 100)}": (d, l) for d in (2,) for l in (0, 0.25, 0.5, 0.75, 1.0)}
ANALYTIC.update({f"dial_d1_l{int(l * 100)}": (1, l) for l in (0, 0.5, 1.0)})
HAND = {"hand_l0": 0.0, "hand_affine50": 0.5, "hand_affine100": 1.0}


def analytic_cell(name):
    d, lam = ANALYTIC[name]
    w = analytic.dial(lam, d=d, iid=1.0, seed=100 + int(lam * 100) + d)
    col = w.collect(6000, T=5)
    pop = w.population(256, 64)
    nominal = [dict(kind="affine", g=lam, scale=0.609)]
    c = PL.Cell(name, col, pop, list(range(d)), {"all": list(range(d))}, w.axes, 1.0, w.gain, nominal=nominal,
                true_cmd=w.channels, iid=1.0)
    return c


def hand_cell(name):
    w = cells.STAGE0[name]()
    col = worlds.load(f"{DATA}/{name}_col_8192x5.npz")
    pop = worlds.load(sorted(glob.glob(f"{DATA}/{name}_pop_512x*.npz"))[0])
    return cells.pipeline_cell(w, col, pop)


def run(kind, names):
    for name in names:
        t = time.time()
        cell = analytic_cell(name) if kind == "analytic" else hand_cell(name)
        cfg = CFG_AN if kind == "analytic" else CFG_HAND
        RN.run_seeds(cell, ARMS, SEEDS, cfg, "s1")
        print(name, "done", round(time.time() - t), "s", flush=True)


def report():
    rows = {}
    notes = """
## Setup

- Analytic dial (`cwk/analytic.py`): x_{t+1} = 0.5 x_t + m(c_t) { g a + sigma eps d (1 - lambda + lambda a) } (the dial of App. D eq. 39 per coordinate), eps = tanh of a Gaussian driver, d = (+1, -1),
  actions uniform on [-1, 1]^d, an exogenous context c_t modulating the scale (m = 1 + 0.3 tanh c), sigma 0.609, g 0.383; iid driver (fresh noise only); 6000 factual episodes x 5 transitions, evaluation
  population 256 x 64 x 3 queries; d = 1 (scalar, the case in which rank-one PCK and the signed location-scale rule coincide) and d = 2. Only 6 of the 8 planned analytic cells were run
  (d = 2: lambda 0, 0.5, 1; d = 1: lambda 0, 0.5, 1) because of compute contention.
- HandReach dial cells: the authors' hand (lambda = 0, 512 parents x 128 draws), hand_affine50 and hand_affine100 (512 x 64), 8192 factual episodes x 5 transitions, state level, iid (v = 1).
- Pipeline as in Stage 2 (see STAGE2.md): MLP mean predictor with 2-fold cross-fitting (HandReach: width 384, 25 epochs, batch 512; analytic: width 256, 20 epochs), kernel heads on the scored coordinates
  (wrist; analytic: all outcome coordinates), 5 fit seeds, whole-parent x seed paired bootstrap. Baselines as listed in STAGE2.md; `indist_gain` uses the exact fault type of the cell (lambda, sigma).
- Rows whose `pck_rank1` is `n/a` were produced before a fix of the rank-one head (random restarts, see `cwk/kernels.py`) and the arm was removed; the d = 1 cells do not need it (R = 1 is the signed scale).
"""
    md = ["# Stage 1: honesty check on rank-one cells\n", notes,
          "R^2 (%) of the counterfactual on the scored group, 5 fit seeds, whole-parent x seed bootstrap 95% CI. "
          "`pck_pol` = polarization-only PCK. Differences are paired (same fit seeds, same bootstrap resamples).\n"]
    for kind, table in (("analytic", ANALYTIC), ("hand", HAND)):
        md.append(f"\n## {'Analytic dial (App. D eq. 39 analogue, d = 1 or 2 coordinates)' if kind == 'analytic' else 'HandReach dial cells (authors configurations, state level)'}\n")
        for name in table:
            raws = [json.load(open(p)) for p in sorted(glob.glob(f"{RN.RAW}/s1_{name}_s*.json"))]
            if len(raws) < 5:
                md.append(f"\n_{name}: {len(raws)}/5 seeds complete_\n")
                continue
            cell = analytic_cell(name) if kind == "analytic" else hand_cell(name)
            det = RN.deterministic(cell)
            if not all(r.get("rank1_version") == 2 for r in raws):      # stale rank-one arm (pre multi-start fix)
                for r in raws:
                    r["losses"].pop("pck_rank1", None)
            res = RN.CellResult(cell, raws, det, 5)
            for g in cell.groups:
                tb = res.table(g)
                md.append(f"\n**{name} / {g}** (rho = {res.rho[g]:.2f})\n")
                md.append("| arm | R^2 % | 95% CI |\n|---|---|---|")
                order = ["oracle", "copy", "rr_exact", "oracle_kernel", "trained", "rr_learned"] + ARMS
                for a in order:
                    if a not in tb:
                        continue
                    r = tb[a]
                    md.append(f"| {a} | {100 * r['r2']:.1f} | [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}] |")
                for other in ("pck_rank1", "ls_signed", "ls_signed_nug", "rr_learned", "ls_unsigned"):
                    if other not in res.L[g]:
                        continue
                    dd = res.diff(g, "pck_pol", other)
                    md.append(f"\nPCK-pol minus {other}: {100 * dd['diff']:+.1f} points, paired 95% CI [{100 * dd['ci95'][0]:+.1f}, {100 * dd['ci95'][1]:+.1f}]")
                rows[f"{name}/{g}"] = {a: tb[a] for a in tb}
                rows[f"{name}/{g}"]["_diff_pck_minus_ls_signed"] = res.diff(g, "pck_pol", "ls_signed")
                rows[f"{name}/{g}"]["_nll"] = {k: float(np.mean([d[k] for d in res.diag if k in d])) for k in res.diag[0] if k.startswith("nll_")}
                md.append("")
    # summary table of the tie claim
    head = ["\n## Summary: polarization-only PCK against the signed location-scale rules\n",
            "Differences in R^2 points (paired bootstrap 95% CI). `ls_signed`: per-coordinate signed affine scale with ridge 1e-3 (the reviewer's "
            "rule, applied coordinatewise); `ls_signed_nug`: same with the fitted nugget as the ridge; `pck_rank1`: the vector-valued rank-one "
            "rule (one signed scale per action, shared by all coordinates), i.e. PCK with R = 1; `pck_pol`: R = 4. "
            "Scalar cells (d1) are the case in which the signed location-scale rule and rank-one PCK coincide.\n",
            "| cell | rho | pck_pol | pck_rank1 | ls_signed | ls_signed_nug | ls_unsigned | rr_learned | pol - ls_signed | pol - ls_signed_nug | pol - rank1 | pol - rr_learned |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind, table in (("analytic", ANALYTIC), ("hand", HAND)):
        for name in table:
            raws = [json.load(open(p_)) for p_ in sorted(glob.glob(f"{RN.RAW}/s1_{name}_s*.json"))]
            if len(raws) < 5:
                continue
            cell = analytic_cell(name) if kind == "analytic" else hand_cell(name)
            if not all(r.get("rank1_version") == 2 for r in raws):
                for r in raws:
                    r["losses"].pop("pck_rank1", None)
            res = RN.CellResult(cell, raws, RN.deterministic(cell), 5)
            g = list(cell.groups)[0]
            def rr(a):
                return f"{100 * res.r2(g, a)['r2']:.1f}" if a in res.L[g] else "n/a"
            def dd(a, b):
                if a not in res.L[g] or b not in res.L[g]:
                    return "n/a"
                d = res.diff(g, a, b)
                return f"{100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
            head.append(f"| {name} | {res.rho[g]:.2f} | {rr('pck_pol')} | {rr('pck_rank1')} | {rr('ls_signed')} | {rr('ls_signed_nug')} | {rr('ls_unsigned')} | {rr('rr_learned')} | "
                        f"{dd('pck_pol', 'ls_signed')} | {dd('pck_pol', 'ls_signed_nug')} | {dd('pck_pol', 'pck_rank1')} | {dd('pck_pol', 'rr_learned')} |")
    md[2:2] = head
    open(f"{OUT}/STAGE1.md", "w").write("\n".join(md))
    json.dump(rows, open(f"{OUT}/stage1.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        kind = sys.argv[2]
        names = sys.argv[3:] or list(ANALYTIC if kind == "analytic" else HAND)
        run(kind, names)
    else:
        report()
