"""Shared pieces of the paper experiments (protocol: results/paper/PROTOCOL.md)."""
import sys, os, glob, json, time, dataclasses
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import cells, worlds, runner as RN, metrics as M

DATA = "/home/user/world_models/results/cwk/data"          # large arrays (gitignored)
PAPER = "/home/user/world_models/results/paper"
RAWP = f"{PAPER}/raw"
LOGS = f"{PAPER}/logs"
for _d in (RAWP, LOGS, DATA):
    os.makedirs(_d, exist_ok=True)

# Stage 2 protocol (results/cwk/STAGE2.md): mean predictor 3 x 384 SiLU, 20 epochs, batch 512; kernel heads 3000 steps;
# 5 fit seeds; 512 parents x 64 draws x 3 queries; 2-fold cross-fitting; 8 k-means context bins.
CFG = dict(epochs=20, batch=512, steps=3000, width=384, n_seeds=5, threads=2, bins=8, cvae_steps=4000,
           eb_Ks=(16, 32), eb_restarts=2, eb_steps=3000, idt_steps=3000, flow_steps=1500, flow_S=4096,
           npmle_M=2000, npmle_iters=150, npmle_n=20000)
COL_N, POP_N, POP_E, DEV_N, DEV_E, FD_H = 16384, 512, 64, 64, 32, 0.5
BOOT = 2000

STAGE2_NINE = ["E1_nu50_gain_iid50", "E1_nu80_gain_iid50", "E2_cup_iid50", "E3_fetch_iid50", "E1_nu50_sat_iid50",
               "E1_nu50_deadzone_iid50", "E1_nu50_gain_iid100", "E2_cup_iid100", "E3_fetch_iid100"]
TIDY = ["rr_learned"]          # recorded by every run, listed for documentation only


def get_world(name):
    if name in cells.STAGE2:
        return cells.STAGE2[name]()
    if name in cells.STAGE0:
        return cells.STAGE0[name]()
    if name in cells.UPSTREAM:
        return cells.UPSTREAM[name]()
    if name in cells.ALL_PAPER:
        return cells.ALL_PAPER[name]()
    if name == "SEALED":
        return cells.sealed_cell()
    raise KeyError(name)


def is_new(name):
    """Cells of the paper design carry finite-difference Jacobians (structural oracle)."""
    return name in cells.INFOLIM or name in cells.MISALIGN or name == "SEALED"


def paths(name, w=None):
    g = lambda pat: (glob.glob(f"{DATA}/{name}_{pat}.npz") or [None])[0]
    return dict(col=g("col_*"), pop=g("pop_*"), dev=g("dev_*"))


def ensure_data(name, col=True, pop=True, dev=True, log=print):
    """Generate (and cache in results/cwk/data) the factual collection, the evaluation population and the development
    population of a cell; the population of the paper's own cells also stores the structural linearisation (fd)."""
    w = get_world(name)
    p = paths(name)
    t0 = time.time()
    if col and p["col"] is None:
        d = worlds.collect(w, COL_N, T=5, verbose=False)
        worlds.save(f"{DATA}/{name}_col_{COL_N}x5.npz", d)
        log(f"[data] {name} collection {COL_N} episodes {time.time() - t0:.0f}s")
    if pop and p["pop"] is None:
        d = worlds.population(w, POP_N, POP_E, fd_h=FD_H if is_new(name) else None)
        worlds.save(f"{DATA}/{name}_pop_{POP_N}x{POP_E}.npz", d)
        log(f"[data] {name} population {POP_N}x{POP_E} {time.time() - t0:.0f}s")
    if dev and p["dev"] is None:
        w2 = dataclasses.replace(w, seed_offset=w.seed_offset + 5_000_000)
        d = worlds.population(w2, DEV_N, DEV_E)
        worlds.save(f"{DATA}/{name}_dev_{DEV_N}x{DEV_E}.npz", d)
        log(f"[data] {name} development {DEV_N}x{DEV_E} {time.time() - t0:.0f}s")


def load_cell(name, need_dev=True):
    w = get_world(name)
    p = paths(name)
    col, pop = worlds.load(p["col"]), worlds.load(p["pop"])
    dev = worlds.load(p["dev"]) if (need_dev and p["dev"]) else None
    return w, cells.pipeline_cell(w, col, pop, dev)


def boot_weights(N, seed=2026, B=BOOT):
    return np.random.default_rng(seed).multinomial(N, np.full(N, 1 / N), size=B)


def result_for(name, tag, n_seeds=5, raw=RAWP, extras=True, boots=BOOT):
    """CellResult of a cell from the cached per-seed files (None if incomplete); 2000 whole-parent x seed bootstrap resamples."""
    raws = [json.load(open(p)) for p in sorted(glob.glob(f"{raw}/{tag}_{name}_s*.json"))]
    if len(raws) < n_seeds:
        return None, None
    raws = raws[:n_seeds]
    w, cell = load_cell(name)
    det = RN.deterministic(cell, extras=extras, log=lambda *a, **k: None)
    arms = set.intersection(*[set(r["losses"]) for r in raws])
    raws = [dict(r, losses={a: r["losses"][a] for a in r["losses"] if a in arms}) for r in raws]
    cell.pop = dict(cell.pop, weights=boot_weights(np.asarray(cell.pop["truth"]).shape[0], B=boots))
    return cell, RN.CellResult(cell, raws, det, n_seeds)


def fmt(r, scale=100):
    return f"{scale * r['r2']:.1f} [{scale * r['ci95'][0]:.1f}, {scale * r['ci95'][1]:.1f}]"


def run_cell(name, arms, seeds, tag, cfg=None, extra_cfg=None, log=print):
    cfg = dict(cfg or CFG, **(extra_cfg or {}))
    ensure_data(name, log=log)
    w, cell = load_cell(name)
    t = time.time()
    RN.run_seeds(cell, arms, seeds, cfg, tag, log=log, raw=RAWP)
    log(f"[{tag}] {name} seeds {list(seeds)} done {time.time() - t:.0f}s")


def cli(argv, default_seeds=5):
    """Minimal option parsing: positional cell names, --seeds N, --arms a,b, --tag T, --smoke."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--seeds", type=int, default=default_seeds)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, no cached results (smoke test only)")
    return ap.parse_args(argv)


SMOKE_CFG = dict(epochs=8, width=64, batch=128, steps=100, n_seeds=5, threads=1, bins=2, cvae_steps=50, eb_steps=60, eb_Ks=(8,),
                 eb_restarts=1, idt_steps=60, idt_restarts=1, flow_steps=60, flow_S=256, npmle_M=200, npmle_iters=10, npmle_n=2000)


def smoke_cell(name, arms, tag, seeds=(0,), extra_cfg=None, log=print):
    """Tiny-size smoke run of a cell (600 episodes, 12 parents x 8 draws, dev 8 x 8; temporary files only)."""
    import tempfile
    w = get_world(name)
    col = worlds.collect(w, 600)
    pop = worlds.population(w, 12, 8, fd_h=FD_H if is_new(name) else None)
    dev = worlds.population(dataclasses.replace(w, seed_offset=w.seed_offset + 5_000_000), 8, 8)
    cell = cells.pipeline_cell(w, col, pop, dev)
    tmp = tempfile.mkdtemp()
    cfg = dict(SMOKE_CFG, **(extra_cfg or {}))
    RN.run_seeds(cell, arms, seeds, cfg, tag, log=log, raw=tmp)
    return w, cell, tmp
