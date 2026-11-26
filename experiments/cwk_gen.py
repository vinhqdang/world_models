"""Generate a factual collection and/or a counterfactual population for a named cell.
usage: python3 experiments/cwk_gen.py CELL pop PARENTS DRAWS | CELL col EPISODES [T]"""
import sys, time, os
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import worlds, cells

OUT = "/home/user/world_models/results/cwk/data"
os.makedirs(OUT, exist_ok=True)

def get_cell(name):
    for table in (cells.STAGE0, getattr(cells, "STAGE2", {})):
        if name in table:
            return table[name]()
    raise KeyError(name)

if __name__ == "__main__":
    name, kind = sys.argv[1], sys.argv[2]
    w = get_cell(name)
    t = time.time()
    if kind == "dev":
        import dataclasses
        n, e = int(sys.argv[3]), int(sys.argv[4])
        w2 = dataclasses.replace(w, seed_offset=w.seed_offset + 5_000_000)
        d = worlds.population(w2, n, e, verbose=True)
        worlds.save(f"{OUT}/{name}_dev_{n}x{e}.npz", d)
    elif kind == "pop":
        n, e = int(sys.argv[3]), int(sys.argv[4])
        d = worlds.population(w, n, e, verbose=True)
        worlds.save(f"{OUT}/{name}_pop_{n}x{e}.npz", d)
    else:
        n = int(sys.argv[3]); T = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        d = worlds.collect(w, n, T=T, verbose=True)
        worlds.save(f"{OUT}/{name}_col_{n}x{T}.npz", d)
    print(name, kind, "done in", round(time.time() - t), "s", flush=True)
