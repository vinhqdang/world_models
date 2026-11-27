"""Aggregation of the paper experiments into results/paper/R*.md and R*.json (2000 whole-parent x seed paired bootstrap resamples).
usage: python3 experiments/paper_report.py r0|r1|r2|r3|r4|r5|r6|all"""
from paper_common import *
import math

PRIM = None


def md_table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def r2_row(res, g, arms):
    out = []
    for a in arms:
        out.append(fmt(res.r2(g, a)) if a in res.L[g] else "n/a")
    return out


def dstr(dd):
    return f"{100 * dd['diff']:+.2f} [{100 * dd['ci95'][0]:+.2f}, {100 * dd['ci95'][1]:+.2f}]"


def pooled(res, g, a):
    return 100 * res.r2(g, a)["r2"] if a in res.L[g] else float("nan")


def cell_block(name, cell, res, arms_order, extra=None):
    g = cell.primary
    lines = [f"\n### {name} (primary group {g}; interaction ratio rho {res.rho[g]:.2f})\n"]
    arms = [a for a in arms_order if a in res.L[g]] + [a for a in res.L[g] if a not in arms_order]
    rows = [(a + ("*" if a in RN.PRIVILEGED else ""), fmt(res.r2(g, a))) for a in arms]
    lines += md_table(["arm", "R^2 % [95% CI]"], rows)
    return lines


ORDER = ["oracle", "copy", "rr_exact", "oracle_kernel", "oracle_struct_bayes", "oracle_quad_ex", "oracle_kernel_lm", "oracle_quad", "hh_nug_ref",
         "trained", "rr_learned", "ls_signed_nug", "hh_global", "hh_binned", "hh_binned_nug", "hh_basis_x", "hh_basis_abs", "hh_basis_x_x3",
         "hh_basis_cubic", "glr_iso", "glr_iso_lamo", "rc_eb_gmm", "rc_npmle", "rc_flow", "idt", "cvae", "pck_pol"]


def diag_mean(raws_diag, key):
    v = [d[key] for d in raws_diag if key in d and not isinstance(d[key], (dict, list))]
    return float(np.mean(v)) if v else float("nan")


# ----------------------------------------------------------------------------------------------------------------------- R1
def report_r1():
    md, js = ["# R1: five-seed confirmation on the nine Stage 2 cells\n",
              "Counterfactual R^2 (%), whole-parent x fit-seed paired bootstrap (2000 resamples). * privileged. Protocol: results/paper/PROTOCOL.md.\n"], {}
    rows1, rows2 = [], []
    for name in STAGE2_NINE:
        cell, res = result_for(name, "r1")
        if res is None:
            md.append(f"\n_{name}: incomplete_\n")
            continue
        g = cell.primary
        md += cell_block(name, cell, res, ORDER)
        ref = "oracle_kernel_lm"
        row = [name]
        flags = []
        for a in ("hh_binned_nug", "rc_eb_gmm", "oracle_quad"):
            dd = res.diff(g, a, ref)
            row.append(dstr(dd))
            flags.append(abs(100 * dd["diff"]) <= 1.0)
        row.append("yes" if all(flags) else "no")
        rows1.append(row)
        js[name] = dict(P1_flags=flags, ceiling=dict(mean_effect=pooled(res, g, "oracle_kernel") - pooled(res, g, "oracle_kernel_lm"),
                                                       kernel_effect=pooled(res, g, "oracle_kernel_lm") - pooled(res, g, "hh_binned_nug")),
                        r2={a: pooled(res, g, a) for a in res.L[g]}, nug_c=diag_mean(res.diag, "nug_c_hh_binned_nug"))
        rows2.append([name, f"{pooled(res, g, 'oracle_kernel'):.1f}", f"{pooled(res, g, 'oracle_kernel_lm'):.1f}", f"{pooled(res, g, 'hh_binned_nug'):.1f}",
                      f"{pooled(res, g, 'hh_binned'):.1f}", f"{pooled(res, g, 'hh_global'):.1f}",
                      f"{js[name]['ceiling']['mean_effect']:.1f}", f"{js[name]['ceiling']['kernel_effect']:.1f}"])
        for a in BASIS_ARMS:
            if a in res.L[g]:
                js[name][f"cond_{a}"] = diag_mean(res.diag, f"cond_{a}")
    md += ["\n## P1 (saturation): difference to oracle_kernel_lm* (points)\n"]
    md += md_table(["cell", "hh_binned_nug", "rc_eb_gmm", "oracle_quad*", "all within 1.0"], rows1)
    md += ["\n## Ceiling decomposition (Prop 6): mean effect = oracle_kernel* - oracle_kernel_lm*, kernel effect = oracle_kernel_lm* - hh_binned_nug\n"]
    md += md_table(["cell", "oracle_kernel*", "oracle_kernel_lm*", "hh_binned_nug", "hh_binned", "hh_global", "mean effect", "kernel effect"], rows2)
    js["P1_overall"] = bool(all(all(v["P1_flags"]) for k, v in js.items() if isinstance(v, dict) and "P1_flags" in v) and len(rows1) == len(STAGE2_NINE))
    md.append(f"\nP1 verdict: {'confirmed' if js['P1_overall'] else 'not confirmed'} on {len(rows1)} cells\n")
    open(f"{PAPER}/R1.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R1.json", "w"), indent=1)
    return js


BASIS_ARMS = ["hh_basis_x", "hh_basis_abs", "hh_basis_x_x3", "hh_basis_cubic"]


# ----------------------------------------------------------------------------------------------------------------------- R2
def r2_names():
    from paper_r2 import ENVS, SEEDS5, SWEEP
    return [f"{e}_{c}" for e in ENVS for c in SEEDS5 + SWEEP] + (["SEALED"] if os.path.exists(cells.SEALED_JSON) else [])


def report_r2():
    from paper_r2 import n_seeds, ENVS
    md, js, R = ["# R2: information-limited cells (ladder)\n", "Counterfactual R^2 (%) on the primary group; * privileged; deterministic arms oracle*, rr_exact*, oracle_kernel*, "
                 "oracle_quad_ex*, oracle_struct_bayes*.\n"], {}, {}
    for name in r2_names():
        cell, res = result_for(name, "r2", n_seeds(name))
        if res is None:
            md.append(f"\n_{name}: incomplete_\n")
            continue
        R[name] = (cell, res)
        g = cell.primary
        md += cell_block(name, cell, res, ORDER)
        sd = getattr(cell, "struct_diag", {}).get(g)
        js[name] = dict(r2={a: pooled(res, g, a) for a in res.L[g]}, oracle_r2=pooled(res, g, "oracle"), rho=res.rho[g], struct=sd,
                        glr_lambda=[d.get("glr_lambda") for d in res.diag], glr_lambda_oracle=[d.get("glr_lambda_oracle") for d in res.diag],
                        nug_c=diag_mean(res.diag, "nug_c_hh_binned_nug"), K_gmm=[d.get("K_rc_eb_gmm") for d in res.diag])
        if sd:
            md.append(f"\nstructural linearisation R^2 {100 * sd['lin_r2']:.2f}%, mean misfit sd {sd['s_mean']:.4f}\n")
    # P2: Gaussian control
    rows, ok2 = [], []
    for e in ENVS:
        n = f"{e}_G"
        if n in R:
            cell, res = R[n]
            g = cell.primary
            dd = res.diff(g, "rc_eb_gmm", "hh_binned_nug")
            ok2.append(abs(100 * dd["diff"]) <= 1.0)
            rows.append([n, dstr(dd), "yes" if ok2[-1] else "no"])
    md += ["\n## P2 (Gaussian control): rc_eb_gmm - hh_binned_nug on the G cells\n"] + md_table(["cell", "difference [95% CI]", "|diff| <= 1.0"], rows)
    js["P2"] = dict(flags=ok2, confirmed=bool(ok2 and all(ok2)))
    # P3: headroom vs pi and r - d
    rows, p3 = [], {}
    for e in ENVS:
        seq_pi = [(p, f"{e}_{c}") for p, c in ((0.05, "pi5"), (0.1, "pi10"), (0.2, "P"), (0.5, "pi50"), (1.0, "pi100"))]
        seq_r = [(0, f"{e}_rd"), (1, f"{e}_P"), (2, f"{e}_rd2")]
        def head(n):
            if n not in R:
                return None
            cell, res = R[n]
            dd = res.diff(cell.primary, "rc_eb_gmm", "hh_binned_nug")
            return dd
        hp = [(p, head(n)) for p, n in seq_pi]
        hr = [(k, head(n)) for k, n in seq_r]
        for p, dd in hp:
            if dd:
                rows.append([e, f"pi={p}", "r-d=1", dstr(dd)])
        for k, dd in hr:
            if dd and k != 1:
                rows.append([e, "pi=0.2", f"r-d={k}", dstr(dd)])
        v = [100 * dd["diff"] for p, dd in hp if dd]
        vr = [100 * dd["diff"] for k, dd in hr if dd]
        mono_pi = len(v) == 5 and all(v[i] >= v[i + 1] for i in range(4))
        mono_r = len(vr) == 3 and all(vr[i] <= vr[i + 1] for i in range(2))
        at1 = hp[-1][1]
        p3[e] = dict(headroom_by_pi=v, headroom_by_r_minus_d=vr, monotone_in_pi=bool(mono_pi), monotone_in_r=bool(mono_r),
                     nonneg_at_pi1=bool(at1 is not None and at1["ci95"][1] * 100 >= 0))
    md += ["\n## P3 (headroom): rc_eb_gmm - hh_binned_nug\n"] + md_table(["env", "pi", "r - d", "difference [95% CI]"], rows)
    md.append("\n" + json.dumps(p3))
    js["P3"] = dict(per_env=p3, confirmed=bool(p3 and all(v["monotone_in_pi"] and v["monotone_in_r"] and v["nonneg_at_pi1"] for v in p3.values())))
    for k in ("P2", "P3"):
        md.append(f"\n{k} verdict: {'confirmed' if js[k]['confirmed'] else 'not confirmed'}\n")
    for f in sorted(glob.glob(f"{PAPER}/R2c_*.json")):
        js[os.path.basename(f)[:-5]] = json.load(open(f))
        md.append(f"\n{os.path.basename(f)}: {json.dumps(js[os.path.basename(f)[:-5]])}\n")
    open(f"{PAPER}/R2.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R2.json", "w"), indent=1)
    return js


# ----------------------------------------------------------------------------------------------------------------------- R3
def report_r3():
    from paper_r3 import UP, ARMS
    md, js = ["# R3: upstream cells, no-harm check\n", "|arm - rr_learned| <= 1.5 points for every HH / EB arm (P5).\n"], {}
    rows, flags = [], []
    hh = ["hh_global", "hh_binned", "hh_binned_nug", "rc_eb_gmm", "rc_npmle", "rc_flow", "pck_pol"]
    for n in UP:
        cell, res = result_for(n, "r3")
        if res is None:
            md.append(f"\n_{n}: incomplete_\n")
            continue
        g = cell.primary
        md += cell_block(n, cell, res, ORDER)
        row = [n, f"{pooled(res, g, 'rr_learned'):.1f}"]
        fl = []
        for a in hh:
            if a in res.L[g]:
                dd = res.diff(g, a, "rr_learned")
                row.append(dstr(dd))
                fl.append(abs(100 * dd["diff"]) <= 1.5)
            else:
                row.append("n/a")
        flags.append(all(fl))
        rows.append(row + ["yes" if all(fl) else "no"])
        js[n] = dict(r2={a: pooled(res, g, a) for a in res.L[g]}, within=bool(all(fl)))
    md += ["\n## Differences to rr_learned [95% CI]\n"] + md_table(["cell", "rr_learned"] + hh + ["all within 1.5"], rows)
    js["P5"] = dict(flags=flags, confirmed=bool(flags and all(flags)))
    md.append(f"\nP5 verdict: {'confirmed' if js['P5']['confirmed'] else 'not confirmed'}\n")
    open(f"{PAPER}/R3.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R3.json", "w"), indent=1)
    return js


# ----------------------------------------------------------------------------------------------------------------------- R4
def subset_r2(pk, truth, coords, mask):
    y = np.asarray(truth, np.float64)[:, :, 1:][..., coords]
    m = y.mean((0, 1, 2))
    v = ((y - m) ** 2).sum(-1).mean(1)
    pk = np.asarray(pk)
    return 100 * (1 - pk[:, mask].mean() / v[mask].mean()) if pk.ndim == 2 else float("nan")


def report_r4():
    md, js = ["# R4: sign ambiguity\n"], {}
    f = f"{PAPER}/R4_synth.json"
    if os.path.exists(f):
        S = json.load(open(f))
        rows = []
        for key, runs in S.items():
            for b in ["(1,a)", "(1,|a|)", "(1,a,|a|)", "(1,a,a2,a3)", "mlp_G"]:
                m = lambda k: float(np.mean([r[b][k] for r in runs]))
                rows.append([key, b, f"{m('nll'):.4f}", f"{m('r2'):.1f}", f"{m('r2_cross'):.1f}", f"{m('r2_same'):.1f}",
                             (f"{np.mean([r[b]['cond'] for r in runs]):.1e}" if "cond" in runs[0][b] else "-")])
            rows.append([key, "oracle / rr", "-", f"{np.mean([r['oracle']['r2'] for r in runs]):.1f} / {np.mean([r['rr']['r2'] for r in runs]):.1f}", "-", "-", "-"])
        md += ["## Synthetic d = 1 (mean over seeds): factual NLL, R^2 all / cross-zero / same-side queries\n"]
        md += md_table(["world (s | xi_1 law)", "basis", "held-out NLL", "R^2 all", "R^2 cross-zero", "R^2 same-side", "cond(Phi^T Phi)"], rows)
        js["synth"] = {k: dict(n_seeds=len(v)) for k, v in S.items()}
    rows = []
    for n in ("hand_affine100", "E1_nu50_gain_iid100"):
        raws = [json.load(open(p)) for p in sorted(glob.glob(f"{RAWP}/r4_{n}_s*.json"))]
        if len(raws) < 5:
            md.append(f"\n_{n}: incomplete_\n")
            continue
        w, cell = load_cell(n, need_dev=False)
        truth, pop, g = np.asarray(cell.pop["truth"]), cell.pop, cell.primary
        coords = cell.groups[g]
        A = np.asarray(pop["actions"])[..., cell.axes] / cell.a_lim
        cross = (np.sign(A[:, 1:]) != np.sign(A[:, :1])).any(-1)
        res = {}
        for a in ("hh_basis_x", "hh_basis_abs", "rr_learned", "oracle_kernel_lm"):
            allr = float(np.mean([100 * (1 - np.mean(r["losses"][a][g]) / M.target_variance(truth, coords)) for r in raws]))
            cr = float(np.mean([subset_r2(r["pk_losses"][a][g], truth, coords, cross) for r in raws]))
            sa = float(np.mean([subset_r2(r["pk_losses"][a][g], truth, coords, ~cross) for r in raws]))
            res[a] = dict(r2=allr, r2_cross=cr, r2_same=sa)
        nll = {a: float(np.mean([r["diag"][f"nll_{a}"] for r in raws])) for a in ("hh_basis_x", "hh_basis_abs")}
        res["nll"], res["cross_fraction"] = nll, float(cross.mean())
        res["P4"] = dict(nll_diff=nll["hh_basis_abs"] - nll["hh_basis_x"], cross_loss=res["hh_basis_x"]["r2_cross"] - res["hh_basis_abs"]["r2_cross"])
        js[n] = res
        for a in ("hh_basis_x", "hh_basis_abs", "rr_learned", "oracle_kernel_lm"):
            rows.append([n, a, f"{res[a]['r2']:.1f}", f"{res[a]['r2_cross']:.1f}", f"{res[a]['r2_same']:.1f}", f"{nll[a]:.4f}" if a in nll else "-"])
    if rows:
        md += ["\n## Real cells: psi (1,x) vs (1,|x|)\n"] + md_table(["cell", "arm", "R^2 all", "R^2 cross-zero", "R^2 same-side", "held-out factual NLL"], rows)
    if "hand_affine100" in js:
        p = js["hand_affine100"]["P4"]
        ok = abs(p["nll_diff"]) <= 0.01 and p["cross_loss"] >= 10
        js["P4"] = dict(nll_diff=p["nll_diff"], cross_zero_loss_points=p["cross_loss"], confirmed=bool(ok))
        md.append(f"\nP4 verdict (hand_affine100): NLL difference {p['nll_diff']:+.4f} nats, cross-zero loss {p['cross_loss']:.1f} points: {'confirmed' if ok else 'not confirmed'}\n")
    open(f"{PAPER}/R4.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R4.json", "w"), indent=1)
    return js


# ----------------------------------------------------------------------------------------------------------------------- R5
def report_r5():
    md, js = ["# R5: antisymmetric gauge\n"], {}
    f = f"{PAPER}/R5_synth.json"
    if os.path.exists(f):
        S = json.load(open(f))
        rows = []
        for name, runs in S.items():
            keys = [k for k in runs[0] if k not in ("r_true",) and not k.endswith("_fit")]
            for k in keys:
                rows.append([name, k, f"{np.mean([r[k] for r in runs]):.1f}", f"{np.std([r[k] for r in runs]):.2f}"])
        md += ["## Synthetic worlds (R^2 %, mean and sd over seeds; hh_lift_R<k> = Burer-Monteiro lift rank k)\n"] + md_table(["world", "arm", "mean", "sd"], rows)
        rot = np.mean([r["hh_lift_R2"] for r in S["rotation_plusJ"]])
        rr = np.mean([r["rr"] for r in S["rotation_plusJ"]])
        mis = np.mean([r["hh_lift_R3"] for r in S["misalign_offset"]])
        orc = np.mean([r["oracle"] for r in S["misalign_offset"]])
        js["P6"] = dict(rotation_hh_lift=float(rot), rotation_rr=float(rr), misalign_hh_lift=float(mis), misalign_oracle=float(orc),
                        confirmed=bool(rot < rr and abs(mis - orc) <= 1.0))
        md.append(f"\nP6: rotation world HH lift {rot:.1f} vs RR {rr:.1f}; misalignment (+offset) HH lift {mis:.1f} vs oracle {orc:.1f}: "
                  f"{'confirmed' if js['P6']['confirmed'] else 'not confirmed'}\n")
    for n in ("MIS_E2o", "MIS_E2o_nooff"):
        cell, res = result_for(n, "r5")
        if res is None:
            md.append(f"\n_{n}: incomplete_\n")
            continue
        md += cell_block(n, cell, res, ORDER)
        js[n] = dict(r2={a: pooled(res, cell.primary, a) for a in res.L[cell.primary]})
    open(f"{PAPER}/R5.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R5.json", "w"), indent=1)
    return js


# ----------------------------------------------------------------------------------------------------------------------- R6
def report_r6():
    from paper_r6 import CELLS, ARMS
    md, js, rows = ["# R6: predictor-accuracy sensitivity (mean predictor trained on 512 episodes)\n"], {}, []
    for n in CELLS:
        cell, res = result_for(n, "r6")
        c1, r1 = result_for(n, "r1")
        if res is None:
            md.append(f"\n_{n}: incomplete_\n")
            continue
        g = cell.primary
        md += cell_block(n, cell, res, ORDER)
        js[n] = dict(r2={a: pooled(res, g, a) for a in res.L[g]}, r2_full_predictor=({a: pooled(r1, g, a) for a in r1.L[g]} if r1 else None))
        for a in ("trained", "rr_learned", "hh_binned_nug", "rc_eb_gmm", "oracle_kernel_lm"):
            rows.append([n, a, f"{pooled(res, g, a):.1f}", f"{pooled(r1, g, a):.1f}" if r1 and a in r1.L[g] else "n/a"])
    md += ["\n## Summary: 512-episode predictor vs the full predictor (R1)\n"] + md_table(["cell", "arm", "512 episodes", "16384 episodes (R1)"], rows)
    open(f"{PAPER}/R6.md", "w").write("\n".join(md))
    json.dump(js, open(f"{PAPER}/R6.json", "w"), indent=1)
    return js


def report_r0():
    parts = ["# R0: unit tests\n"]
    log = []
    for f in sorted(glob.glob(f"{LOGS}/R0_*.log")):
        txt = open(f).read()
        log.append(f"===== {os.path.basename(f)} =====\n{txt}")
        parts.append(f"## {os.path.basename(f)}\n```\n{txt.strip()}\n```")
    open(f"{PAPER}/R0.log", "w").write("\n".join(log))
    open(f"{PAPER}/R0.md", "w").write("\n".join(parts))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fn = dict(r0=report_r0, r1=report_r1, r2=report_r2, r3=report_r3, r4=report_r4, r5=report_r5, r6=report_r6)
    for k in (fn if which == "all" else [which]):
        try:
            fn[k]()
            print(k, "written", flush=True)
        except Exception as e:  # incomplete stages must not stop the others
            print(k, "skipped:", type(e).__name__, e, flush=True)
