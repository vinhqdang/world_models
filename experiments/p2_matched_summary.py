import json, numpy as np
from scipy import stats
def ci(x):
    x = np.array([v for v in x if v == v], float)
    if len(x) == 0: return "n/a"
    if len(x) < 2: return f"{x.mean():.3g}"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return f"{x.mean():.3g} ± {h:.2g}"
out = []
for name in ("lorenz", "vdp"):
    M = [json.loads(l) for l in open(f"/home/user/world_models/results/p2/{name}_matched.jsonl")]
    T = [json.loads(l) for l in open(f"/home/user/world_models/results/p2/{name}_test.jsonl")]
    seeds = sorted({r["seed"] for r in M})
    T = [r for r in T if r["seed"] in seeds]
    out.append(f"\n### {name}: compute-matched controls (seeds {seeds}; SCN uses 10000 optimiser steps in total)\n")
    out.append("| method | VPT | blow-up frac | energy dist. | off-manifold | one-step RMSE |"); out.append("|---|---|---|---|---|---|")
    for lab, R in [("scn (10k)", [r for r in T if r["method"] == "scn"]), ("iso_10k", [r for r in M if r["method"] == "iso_10k"]),
                   ("normal_10k", [r for r in M if r["method"] == "normal_10k"]), ("ss_10k", [r for r in M if r["method"] == "ss_10k"])]:
        g = lambda k: ci([r.get(k, float("nan")) for r in R])
        out.append(f"| {lab} | {g('vpt')} | {g('blow')} | {g('edist')} | {g('offman')} | {g('onestep')} |")
    sc = {r["seed"]: r["vpt"] for r in T if r["method"] == "scn"}
    for m in ("iso_10k", "normal_10k", "ss_10k"):
        b = {r["seed"]: r["vpt"] for r in M if r["method"] == m}
        d = np.array([sc[s] - b[s] for s in seeds if s in b])
        if len(d) > 1:
            h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); out.append(f"\n- paired VPT scn - {m}: {d.mean():+.1f} ± {h:.1f} (n={len(d)})")
txt = "\n".join(out); print(txt); open("/home/user/world_models/results/p2/tables_matched.md", "w").write(txt)
