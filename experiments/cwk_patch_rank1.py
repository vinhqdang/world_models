"""Recompute only the pck_rank1 arm of cached seeds produced before the rank-one multi-start fix (the cross-fitted
predictor is deterministic given the seed, so the paired comparison is unchanged).
usage: python3 experiments/cwk_patch_rank1.py s1|s2 CELL..."""
import sys, json, os
sys.path.insert(0, "/home/user/world_models"); sys.path.insert(0, "/home/user/world_models/experiments")
from cwk import pipeline as PL, runner as RN
stage = sys.argv[1]
for name in sys.argv[2:]:
    if stage == "s1":
        import cwk_stage1 as S
        cell = S.analytic_cell(name) if name in S.ANALYTIC else S.hand_cell(name)
        cfg = S.CFG_AN if name in S.ANALYTIC else S.CFG_HAND
    else:
        import cwk_stage2 as S
        w, cell = S.load_cell(name)
        cfg = S.CFG
    for seed in range(5):
        path = f"{RN.RAW}/{stage}_{name}_s{seed}.json"
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        if r.get("rank1_version") == 2:
            continue
        res = PL.run_seed(cell, seed, ["pck_rank1"], cfg, log=print)
        r["losses"]["pck_rank1"] = {g: v.tolist() for g, v in res["losses"]["pck_rank1"].items()}
        r["rank1_version"] = 2
        json.dump(r, open(path, "w"))
        print("patched", path, flush=True)
