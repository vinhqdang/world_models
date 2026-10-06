"""Aggregate P1 pilot results: per-method means with seed-bootstrap CIs and paired differences vs plain MLE."""
import json, glob, sys, collections
import numpy as np

rng = np.random.RandomState(0)


def boot(x, B=4000):
    x = np.asarray(x)
    idx = rng.randint(0, len(x), (B, len(x)))
    bm = x[idx].mean(1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += json.load(open(f))
    return rows


def table(rows, title, base="mle", metrics=("nregret", "cerr_off", "cerr_on", "sens_off", "cos_off", "mse")):
    g = collections.defaultdict(dict)
    for r in rows:
        g[r["method"]][r["seed"]] = r
    seeds = sorted(set.intersection(*[set(v) for v in g.values()]))
    out = [f"### {title}  (n seeds = {len(seeds)})", "",
           "| method | lam | norm. regret [95% CI] | paired d vs " + base + " [95% CI] | wins/n | contrast err off-support | contrast err on-support | sens. ratio off | cos off | 1-step MSE |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    b = np.array([g[base][s]["nregret"] for s in seeds])
    for m in sorted(g, key=lambda k: np.mean([g[k][s]["nregret"] for s in seeds])):
        v = np.array([g[m][s]["nregret"] for s in seeds])
        mu, lo, hi = boot(v)
        d = v - b
        dm, dlo, dhi = boot(d)
        lam = g[m][seeds[0]].get("lam")
        f = lambda k: np.mean([g[m][s][k] for s in seeds])
        out.append(f"| {m} | {lam if m in ('idm','idm_delta','nce','ridm') else ''} | {mu:.3f} [{lo:.3f},{hi:.3f}] | "
                   f"{'' if m == base else f'{dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]'} | {'' if m == base else f'{int((d < 0).sum())}/{len(d)}'} | "
                   f"{f('cerr_off'):.3f} | {f('cerr_on'):.3f} | {f('sens_off'):.2f} | {f('cos_off'):.3f} | {f('mse'):.5f} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    txt = ["# P1 pilot tables (eval seeds 0-9; dev seeds 900-902 used only for lambda selection)", ""]
    main = load("results/p1/eval/main_*.json")
    if main:
        txt.append(table(main, "Main: sigma_eps=0.05, N=3000"))
    for s in ("0.02", "0.15", "0.4"):
        rows = load(f"results/p1/eval/sw_{s}.json")
        if rows:
            txt.append(table(rows, f"Excitation sweep: sigma_eps={s}, N=3000"))
    open("results/p1/tables.md", "w").write("\n".join(txt))
    print("\n".join(txt))
