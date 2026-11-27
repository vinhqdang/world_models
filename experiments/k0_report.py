"""Aggregate K0 results -> results/k0/K0.md"""
import json, glob, numpy as np
from scipy import stats
D = "/home/user/world_models/results/k0/"; SET = [("g15", "gated pi=0.15"), ("g30", "gated pi=0.30"), ("tanh", "tanh")]
ARMS = [("idt", "idt"), ("rc_eb_gmm", "rc_eb_gmm"), ("hh_blup", "hh_blup"), ("oracle_blup", "oracle_blup*"), ("exact_bayes", "exact_bayes*"), ("glr_iso", "glr_iso")]
seeds = range(5); rb = np.random.default_rng(12345); NB = 2000
def tci(v):
    v = np.asarray(v); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return m, m - h, m + h, h
out = ["# K0 pre-kill: idt vs rc_eb_gmm on the syn2 generator", "", "Protocol: `results/k0/PROTOCOL.md` (written before any run). Runner: `experiments/k0_run.py`; aggregation: `experiments/k0_report.py`. Counterfactual R^2 (%), 20,000 test points, 5 seeds per setting.", ""]
fails = 0; summ = []; flags = []; notes = []
for key, title in SET:
    R = {s: json.load(open(f"{D}raw/{key}_s{s}.json")) for s in seeds}
    out += [f"## {title}", "", "| arm | mean | sd | " + " | ".join(f"s{s}" for s in seeds) + " |", "|---|---|---|" + "---|" * 5]
    for a, nm in ARMS:
        v = [R[s][a] for s in seeds]; out.append(f"| {nm} | {np.mean(v):.2f} | {np.std(v, ddof=1):.2f} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    diff = np.array([R[s]["idt"] - R[s]["rc_eb_gmm"] for s in seeds]); m, lo, hi, h = tci(diff)
    # paired bootstrap over test points, within seed, same indices for both arms
    Z = {s: np.load(f"{D}raw/{key}_s{s}_sq.npz") for s in seeds}; nt = len(Z[0]["SST"]); bs = np.empty(NB)
    # SST per point is |tgt-mean|^2 around the full-test mean; the denominator uses the resampled sum of these
    for b in range(NB):
        ds = []
        for s in seeds:
            i = rb.integers(0, nt, nt); den = Z[s]["SST"][i].sum()
            ds.append(100 * (Z[s]["rc_eb_gmm"][i].sum() - Z[s]["idt"][i].sum()) / den)
        bs[b] = np.mean(ds)
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    fail = (m < 1.5) or (lo <= 0 <= hi) or (blo <= 0 <= bhi); fails += fail
    out += ["", f"idt - rc_eb_gmm per seed: " + ", ".join(f"{x:+.2f}" for x in diff), f"- mean {m:+.2f}; t-interval (4 df) [{lo:+.2f}, {hi:+.2f}]; pooled paired bootstrap over test points [{blo:+.2f}, {bhi:+.2f}] ({NB} replicates)", f"- setting verdict: {'FAIL (mean < 1.5 or an interval includes 0)' if fail else 'pass'}"]
    out.append(f"- t-interval alone: {'includes 0' if lo <= 0 <= hi else 'excludes 0'}; mean >= 1.5: {m >= 1.5}")
    summ.append((title, m, lo, hi, blo, bhi, fail))
    # other contrasts and sanity
    for a in ("hh_blup", "oracle_blup", "glr_iso"):
        d2 = np.array([R[s]["idt"] - R[s][a] for s in seeds]); mm, l2, h2, _ = tci(d2); out.append(f"- idt - {a}: {mm:+.2f} [{l2:+.2f}, {h2:+.2f}]")
    bay = np.array([R[s]["exact_bayes"] for s in seeds]); _, _, _, hb = tci(bay)
    for a in ("idt", "rc_eb_gmm"):
        dd = np.array([R[s][a] - R[s]["exact_bayes"] for s in seeds]); mm, l2, h2, _ = tci(dd)
        bad = (mm > hb) or (l2 > 0); out.append(f"- sanity {a} - exact_bayes*: {mm:+.2f} [{l2:+.2f}, {h2:+.2f}] (exact Bayes CI half-width over seeds {hb:.2f}): {'FLAG' if bad else 'ok'}")
        if bad: flags.append(f"{title}: {a}")
    out.append(f"- idt held-out factual NLL {np.mean([R[s]['idt_val_nll'] for s in seeds]):.3f}, rc_eb_gmm {np.mean([R[s]['gmm_val_nll'] for s in seeds]):.3f} (mean over seeds; lower is better)")
    out.append(f"- glr_iso lambda chosen per seed: {[R[s]['glr_lambda'] for s in seeds]}; R^2 by lambda (mean): " + ", ".join(f"{l}: {np.mean([R[s]['glr_r2_by_lambda'][l] for s in seeds]):.2f}" for l in ['2', '4', '6', '8', '12']))
    if key == "tanh": out.append("- exact_bayes* reference: 3-D midpoint quadrature in driver space, 64 points/dim, exact tanh-Gaussian density. Resolution check on the first 1000 test points (32/64/96 pts), R^2: " + "; ".join(f"seed {s}: " + ", ".join(f"{v:.3f}" for v in R[s]['exact_bayes_res_check_first1000'].values()) for s in seeds))
    else: out.append("- exact_bayes* reference: pattern enumeration plus midpoint grid (h=0.03) with analytic truncated-normal integration of the offset driver; convergence check at h=0.06 (max |R^2 difference| over seeds): " + f"{max(abs(R[s]['exact_bayes'] - R[s]['exact_bayes_coarse_h0.06']) for s in seeds):.4f}")
    out.append("")
out += ["## Verdict", "", "| setting | mean idt - rc_eb_gmm | t 95% CI | bootstrap 95% CI | fails |", "|---|---|---|---|---|"]
for t, m, lo, hi, blo, bhi, f in summ: out.append(f"| {t} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | [{blo:+.2f}, {bhi:+.2f}] | {'yes' if f else 'no'} |")
out += ["", f"Settings failing: {fails} of 3. **K0 {'FAILS (NO-GO)' if fails >= 2 else 'PASSES'}** (NO-GO when 2 or more settings fail).", "", "Sanity flags (learned arm above exact Bayes*): " + (", ".join(flags) if flags else "none"), ""]
open(D + "K0.md", "w").write("\n".join(out)); print("\n".join(out))
