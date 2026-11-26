"""Run all fit seeds of a cell with per-seed caching and summarise (R^2, bootstrap intervals, paired differences)."""
import json
import os

import numpy as np

from cwk import metrics as M, oracle as OR, pipeline as PL

RAW = "/home/user/world_models/results/cwk/raw"
os.makedirs(RAW, exist_ok=True)

PRIVILEGED = {"oracle", "rr_exact", "oracle_kernel", "rho_ref", "indist_struct", "copy"}


def run_seeds(cell, arms, seeds, cfg, tag, log=print):
    out = []
    for s in seeds:
        path = f"{RAW}/{tag}_{cell.name}_s{s}.json"
        if os.path.exists(path):
            out.append(json.load(open(path)))
            continue
        res = PL.run_seed(cell, s, arms, cfg, log=log)
        res = dict(losses={a: {g: v.tolist() for g, v in gl.items()} for a, gl in res["losses"].items()},
                   diag=res["diag"], seconds=res["seconds"], rank1_version=2)
        json.dump(res, open(path, "w"))
        out.append(res)
    return out


def deterministic(cell):
    """Oracle (no outcome), copy, exact-mean residual reuse and the exact-structure (MC) kriging oracle, per group."""
    truth = np.asarray(cell.pop["truth"])
    K = truth.shape[2]
    res = {}
    for g, idx in cell.groups.items():
        mc = OR.mc_kriging(truth, idx)
        pred = np.array(truth)
        pred[..., idx] = mc
        res[g] = dict(oracle=M.oracle_loss(truth, idx), rr_exact=M.exact_mean_residual_loss(truth, idx),
                      copy=M.parent_loss(np.repeat(truth[:, :, :1], K, 2), truth, idx),
                      oracle_kernel=M.parent_loss(pred, truth, idx))
    return res


class CellResult:
    """Per-group losses of every arm (seeds x parents) with R^2, intervals and paired differences."""

    MERGED = {"pck_pol", "pck_echo", "pck_joint", "hh_global"}   # pck_rank1 (vector signed location-scale) IS a baseline

    def __init__(self, cell, raws, det, n_seeds):
        self.cell = cell
        truth = np.asarray(cell.pop["truth"])
        self.arms = list(raws[0]["losses"].keys())
        self.ev, self.L, self.rho = {}, {}, {}
        for g, idx in cell.groups.items():
            self.ev[g] = M.Evaluator(truth, cell.pop["weights"], idx, n_seeds=n_seeds)
            L = {a: np.array([r["losses"][a][g] for r in raws]) for a in self.arms}
            L.update({a: v[None] for a, v in det[g].items()})
            self.L[g] = L
            self.rho[g] = M.interaction_ratio(truth, idx)
        self.diag = [r["diag"] for r in raws]

    def r2(self, g, a):
        return self.ev[g].r2(self.L[g][a])

    def diff(self, g, a, b):
        return self.ev[g].diff(self.L[g][a], self.L[g][b])

    def best_baseline(self, g, exclude=()):
        """Highest-R^2 non-privileged, non-merged arm (point estimate)."""
        cand = [a for a in self.L[g] if a not in PRIVILEGED and a not in self.MERGED and a not in exclude]
        sc = {a: 1 - self.L[g][a].mean() / self.ev[g].var for a in cand}
        return max(sc, key=sc.get), sc

    def table(self, g):
        return {a: self.r2(g, a) for a in self.L[g]}
