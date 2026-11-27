"""Run all fit seeds of a cell with per-seed caching and summarise (R^2, bootstrap intervals, paired differences)."""
import json
import os

import numpy as np

from cwk import metrics as M, oracle as OR, pipeline as PL

RAW = "/home/user/world_models/results/cwk/raw"
os.makedirs(RAW, exist_ok=True)

PRIVILEGED = {"oracle", "rr_exact", "oracle_kernel", "rho_ref", "indist_struct", "copy"} | PL.PAPER_PRIVILEGED


def run_seeds(cell, arms, seeds, cfg, tag, log=print, raw=None):
    """Per-seed caching in ``raw`` (default results/cwk/raw). A cached seed that lacks some requested arm is completed:
    only the missing arms are run and merged (same seed, same fits)."""
    out = []
    raw = raw or RAW
    os.makedirs(raw, exist_ok=True)
    for s in seeds:
        path = f"{raw}/{tag}_{cell.name}_s{s}.json"
        old = json.load(open(path)) if os.path.exists(path) else None
        todo = [a for a in arms if old is None or a not in old["losses"]]
        if old is not None and not todo:
            out.append(old)
            continue
        res = PL.run_seed(cell, s, todo, cfg, log=log)
        res = dict(losses={a: {g: v.tolist() for g, v in gl.items()} for a, gl in res["losses"].items()},
                   diag=res["diag"], seconds=res["seconds"], rank1_version=2, **({"pk_losses": res["pk_losses"]} if "pk_losses" in res else {}))
        if old is not None:
            old["losses"].update({a: v for a, v in res["losses"].items() if a not in old["losses"]})
            old["diag"].update(res["diag"])
            old["seconds"] += res["seconds"]
            if "pk_losses" in res:
                old.setdefault("pk_losses", {}).update(res["pk_losses"])
            res = old
        json.dump(res, open(path, "w"))
        out.append(res)
    return out


def deterministic(cell, extras=False, log=print):
    """Oracle (no outcome), copy, exact-mean residual reuse and the exact-structure (MC) kriging oracle, per group.
    extras: also the exact-mean quadratic oracle (oracle_quad_ex) and, when the population carries finite-difference
    Jacobians, the structural exact-Bayes oracle (oracle_struct_bayes) with its linearisation R^2 (cell.struct_diag)."""
    truth = np.asarray(cell.pop["truth"])
    K = truth.shape[2]
    res = {}
    cell.struct_diag = {}
    for g, idx in cell.groups.items():
        mc = OR.mc_kriging(truth, idx)
        pred = np.array(truth)
        pred[..., idx] = mc
        res[g] = dict(oracle=M.oracle_loss(truth, idx), rr_exact=M.exact_mean_residual_loss(truth, idx),
                      copy=M.parent_loss(np.repeat(truth[:, :, :1], K, 2), truth, idx),
                      oracle_kernel=M.parent_loss(pred, truth, idx))
        if extras:
            q = np.array(truth)
            q[..., idx] = OR.mc_quadratic(truth, idx)
            res[g]["oracle_quad_ex"] = M.parent_loss(q, truth, idx)
            if "fd_J" in cell.pop and cell.world is not None and g == cell.primary:
                nd = np.asarray(cell.pop["eps"]).shape[-1]
                lw = cell.world.laws
                laws = [dict(kind="tanh")] * nd if lw is None else ([lw] * nd if isinstance(lw, dict) else list(lw))
                pr, dg = OR.struct_bayes(cell.pop, laws, idx)
                sb = np.array(truth)
                sb[..., idx] = pr
                res[g]["oracle_struct_bayes"] = M.parent_loss(sb, truth, idx)
                li = np.array(truth)
                li[..., idx] = dg["lin"]
                var = M.target_variance(truth, idx)
                cell.struct_diag[g] = dict(lin_r2=float(1 - M.parent_loss(li, truth, idx).mean() / var), s_mean=dg["s_mean"])
                log(f"  struct Bayes: linearisation R^2 {cell.struct_diag[g]['lin_r2']:.4f}, mean misfit sd {dg['s_mean']:.4f}")
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
