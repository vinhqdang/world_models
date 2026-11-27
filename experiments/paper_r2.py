"""R2: information-limited cells (E1o HandReach d = 2, E2o Ball-in-Cup d = 2, E3o FetchReach d = 3).
usage: python3 experiments/paper_r2.py gen [CELL ...]          R2a data (collection, population with finite-difference Jacobians, development)
       python3 experiments/paper_r2.py ladder [CELL ...]       R2b baseline ladder (5 seeds for P, T, G, Piid50, SEALED; 3 for the rest)
       python3 experiments/paper_r2.py validate [CELL] [--smoke]   R2c: validate oracle_struct_bayes* against full-simulation quadrature
       python3 experiments/paper_r2.py --smoke                  tiny-size smoke test of data generation, ladder and validation
Cells: <env>_{P,T,G,pi5,pi10,pi50,pi100,rd,rd2,Piid50,Pdata1,Pdata2} for env in E1o, E2o, E3o, and SEALED (after the protocol commit)."""
from paper_common import *

ENVS = ["E1o", "E2o", "E3o"]
SEEDS5 = ["P", "T", "G", "Piid50"]
SWEEP = ["pi5", "pi10", "pi50", "pi100", "rd", "rd2", "Pdata1", "Pdata2"]
LADDER = ["ls_signed_nug", "hh_binned", "hh_binned_nug", "hh_nug_ref", "glr_iso", "glr_iso_lamo", "rc_eb_gmm", "rc_npmle", "rc_flow",
          "cvae", "idt", "oracle_kernel_lm", "oracle_quad"]
TAG = "r2"


def all_cells():
    return [f"{e}_{c}" for e in ENVS for c in SEEDS5 + SWEEP]


NPMLE_CELLS = SEEDS5 + ["SEALED"]      # rc_npmle only on P, T, G, Piid50 and SEALED (compute lever, fixed in the protocol)


def arms_for(name):
    keep_npmle = name == "SEALED" or name.split("_")[-1] in NPMLE_CELLS
    return [a for a in LADDER if a != "rc_npmle" or keep_npmle]


def n_seeds(name):
    return 5 if (name.split("_")[-1] in SEEDS5 or name == "SEALED") else 3


def validate(name, parents=64, grid=11, log=print, smoke=False, w=None, cell=None):
    """R2c. Full-simulation quadrature of the exact posterior mean on a gate-pattern x eps grid (every grid point is a simulator
    call), with an ABC-type Gaussian kernel of width proportional to the grid cell response; compared with (i) the same quadrature
    on the linearised map (y0 + J eps) and (ii) oracle_struct_bayes* (noiseless-limit quadrature on the linearised map)."""
    import itertools
    from cwk import oracle as OR
    if cell is None:
        ensure_data(name, dev=False, log=log)
        w, cell = load_cell(name, need_dev=False)
    pop = cell.pop
    idx = cell.groups[cell.primary]
    nd = np.asarray(pop["eps"]).shape[-1]
    laws = [w.laws] * nd if isinstance(w.laws, dict) else list(w.laws)
    pi = [l.get("pi") for l in laws]
    env = w.make()
    spec = w.sim_spec("population")
    N = min(parents, len(pop["truth"]))
    truth = np.asarray(pop["truth"])[:N]
    K, E = truth.shape[2], truth.shape[1]
    g1 = (np.arange(grid) + 0.5) / grid * 2 - 1
    sims = np.empty((N, K, 0))
    res = dict(quad_sim=np.empty((N, E, K, len(idx))), quad_lin=np.empty((N, E, K, len(idx))))
    from scipy.special import ndtri
    from cwk.oracle import _logrho
    post = np.asarray(pop["posterior"])
    for i in range(N):
        env.reset(spec["seed"] + i)
        for t in range(2):
            env.step(pop["past_actions"][i, t], pop["past_eps"][i, t])
        snap = env.snapshot()
        pats = [sub for k in range(nd + 1) for sub in itertools.combinations(range(nd), k)]
        pts, lps, ysim, ylin = [], [], [], []
        J = np.asarray(pop["fd_J"])[i][:, idx, :]
        y0 = np.asarray(pop["fd_y0"])[i][:, idx]
        for sub in pats:
            lpri = sum(np.log(pi[j]) if j in sub else np.log(1 - pi[j]) for j in range(nd) if laws[j]["kind"] == "gated")
            for u in (itertools.product(g1, repeat=len(sub)) if sub else [()]):
                e = np.zeros(nd)
                e[list(sub)] = u
                lp = lpri + sum(float(_logrho(laws[j], np.array([e[j]]), post[i, j, 0], post[i, j, 1])[0]) for j in sub) \
                    + len(sub) * np.log(2.0 / grid)
                ys = []
                for k in range(K):
                    env.restore(snap)
                    ys.append(env.target(env.query(pop["actions"][i, k], e))[idx])
                pts.append(e), lps.append(lp), ysim.append(np.array(ys)), ylin.append(y0 + np.einsum("kdj,j->kd", J, e))
        pts, lps, ysim, ylin = map(np.array, (pts, lps, ysim, ylin))
        h = (2.0 / grid) * np.abs(J[0]).sum(1).mean() * 0.5           # kernel width: half a grid cell of the executed-action response
        for key, Y in (("quad_sim", ysim), ("quad_lin", ylin)):
            for e_ in range(E):
                yo = truth[i, e_, 0][idx]
                lw = lps - 0.5 * ((Y[:, 0] - yo) ** 2).sum(1) / h ** 2
                wgt = np.exp(lw - lw.max())
                wgt /= wgt.sum()
                res[key][i, e_] = np.einsum("p,pkd->kd", wgt, Y)
        if (i + 1) % 8 == 0:
            log(f"[validate {name}] parent {i + 1}/{N}")
    env.close()
    sub = dict(pop, truth=truth, eps=np.asarray(pop["eps"])[:N], fd_J=np.asarray(pop["fd_J"])[:N], fd_y0=np.asarray(pop["fd_y0"])[:N],
               posterior=post[:N])
    pr, dg = OR.struct_bayes(sub, laws, idx)
    var = M.target_variance(truth, idx)
    out = {}
    for key, pred in (("quad_sim", res["quad_sim"]), ("quad_lin", res["quad_lin"]), ("struct_bayes", pr)):
        full = np.array(truth)
        full[..., idx] = pred
        out[key] = float(100 * (1 - M.parent_loss(full, truth, idx).mean() / var))
    li = np.array(truth)
    li[..., idx] = dg["lin"]
    out["lin_r2_true_eps"] = float(100 * (1 - M.parent_loss(li, truth, idx).mean() / var))
    out["parents"], out["grid"] = N, grid
    if not smoke:
        json.dump(out, open(f"{PAPER}/R2c_{name}.json", "w"), indent=1)
    log(f"[validate {name}] R^2 % {out}")
    return out


if __name__ == "__main__":
    a = cli(sys.argv[1:])
    log = lambda *x, **k: print(*x, flush=True)
    if a.smoke:
        w, cell, tmp = smoke_cell("E1o_P", ["hh_binned_nug", "rc_eb_gmm", "glr_iso", "rc_npmle"], TAG)
        from cwk import runner as RN2
        det = RN2.deterministic(cell, extras=True)
        print("R2 smoke ok; deterministic arms:", list(det[cell.primary]))
        print(validate("E1o_P", parents=2, grid=5, log=log, smoke=True, w=w, cell=cell))
        sys.exit(0)
    cmd, names = a.cells[0], a.cells[1:]
    names = names or (all_cells() + (["SEALED"] if os.path.exists(cells.SEALED_JSON) else []))
    if cmd == "gen":
        for n in names:
            ensure_data(n, log=log)
    elif cmd == "ladder":
        for n in names:
            run_cell(n, arms_for(n), range(a.seeds if a.seeds != 5 else n_seeds(n)), TAG, log=log)
    elif cmd == "validate":
        validate(names[0] if a.cells[1:] else "E1o_P", log=log)
