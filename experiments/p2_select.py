"""Pick hyper-parameters from VALIDATION runs only (val seeds 100+, val initial conditions)."""
import json, sys, numpy as np, collections
name = sys.argv[1]
rows = [json.loads(l) for l in open(f"/home/user/world_models/results/p2/{name}_tune.jsonl")]
agg = collections.defaultdict(list)
for r in rows:
    if r["method"].startswith(("iso@", "normal@", "ss@")): agg[r["method"]].append(r["vpt"])
best = {}
for fam in ("iso", "normal", "ss"):
    c = {k: np.mean(v) for k, v in agg.items() if k.startswith(fam + "@")}
    k = max(c, key=c.get); best[fam] = float(k.split("@")[1]); print(fam, {a: round(b, 1) for a, b in c.items()}, "->", k)
cfg = dict(iso=best["iso"], normal=best["normal"], ss_p=best["ss"])
json.dump(cfg, open(f"/home/user/world_models/results/p2/{name}_selected.json", "w")); print(cfg)
