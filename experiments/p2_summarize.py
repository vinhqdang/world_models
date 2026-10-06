import json, sys, numpy as np, collections
from scipy import stats
rows = []
for name in ("lorenz", "vdp"):
    try: rows += [json.loads(l) for l in open(f"/home/user/world_models/results/p2/{name}_test.jsonl")]
    except FileNotFoundError: pass
def ci(x):
    x = np.array([v for v in x if v == v], float)
    if len(x) == 0: return "n/a"
    if len(x) < 2: return f"{x.mean():.3g}"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.3g} ± {h:.2g}"
out = []
for name in ("lorenz", "vdp"):
    R = [r for r in rows if r["system"] == name]
    if not R: continue
    seeds = sorted({r["seed"] for r in R})
    out.append(f"\n### {name} (test seeds {seeds}, mean ± 95% t-CI over seeds)\n")
    out.append("| method | VPT (steps, thr 0.3) | VPT (Lyapunov times) | blow-up frac (3000 steps) | energy dist. to attractor | off-manifold dist | one-step RMSE |")
    out.append("|---|---|---|---|---|---|---|")
    order = ["tf", "tf+aeproj", "iso", "iso+aeproj", "normal", "ss", "scn", "noisy_oracle@tf_eps", "noisy_oracle@tf_eps/2"]
    for m in order:
        X = [r for r in R if r["method"] == m]
        if not X: continue
        g = lambda k: ci([r.get(k, float("nan")) for r in X])
        ly = ci([r["vpt_lyap"] for r in X]) if "vpt_lyap" in X[0] and name == "lorenz" else "-"
        out.append(f"| {m} | {g('vpt')} | {ly} | {g('blow')} | {g('edist')} | {g('offman')} | {g('onestep')} |")
    # paired differences vs best baseline on VPT
    base = {}
    for m in ("tf", "iso", "normal", "ss", "scn", "iso+aeproj", "tf+aeproj"):
        base[m] = {r["seed"]: r["vpt"] for r in R if r["method"] == m}
    out.append("\nPaired VPT differences (seed-matched), scn minus X:\n")
    for m in ("tf", "iso", "normal", "ss", "iso+aeproj"):
        s = [k for k in base[m] if k in base["scn"]]
        if len(s) >= 2:
            d = np.array([base["scn"][k] - base[m][k] for k in s]); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out.append(f"- scn - {m}: {d.mean():+.2f} ± {h:.2f} (n={len(d)})")
txt = "\n".join(out); print(txt); open("/home/user/world_models/results/p2/tables.md", "w").write(txt)
