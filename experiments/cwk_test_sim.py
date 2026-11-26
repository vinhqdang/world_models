"""Checks that cwk.worlds reproduces the authors' populations (cwm.sims.population) and that snapshot/restore equals
replay. Run: python3 experiments/cwk_test_sim.py"""
import sys, tempfile, json
sys.path.insert(0, "/home/user/world_models")
import numpy as np
from cwk import upstream  # noqa
from cwk import worlds
from cwm import registry
from cwm.sims import population as up_pop

CELLS = {  # name: (upstream config, channels)
    "hand": ("hand", [dict(kind="affine", g=0.0)]),
    "hand_affine50": ("hand_affine50", [dict(kind="affine", g=0.5)]),
    "hand_affine100": ("hand_affine100", [dict(kind="affine", g=1.0)]),
}

def run(name, base, channels, parents=3, fields=4, iid=None):
    from cwm.sims import resolve
    spec = resolve(registry.simulator(base, "population"))
    sc = spec["disturbance_scale"]
    channels = [dict(c, scale=sc) for c in channels]
    iid = spec["benchmark_design"]["iid_fraction"] if iid is None else iid
    w = worlds.World(name, base, channels=channels, iid_fraction=iid)
    mine = worlds.population(w, parents, fields)
    s2 = dict(spec, parents=parents, fields=fields, save_frames=False)
    out = tempfile.mkdtemp() + "/pop"
    up_pop.run(s2, out)
    ref = np.load(out + "/branches.npz")
    err = np.abs(mine["truth"] - ref["truth"]).max()
    print(name, "max |truth - upstream truth| =", err, "eps err", np.abs(mine["eps"][..., 0] - ref["epsilon"]).max())
    return err

if __name__ == "__main__":
    for n, (b, ch) in CELLS.items():
        assert run(n, b, ch) == 0.0
    # iid 0.5 variant against upstream hand_iid50 (offset)
    from cwm.sims import resolve
    spec = resolve(registry.simulator("hand_iid50", "population"))
    w = worlds.World("hand_iid50", "hand_iid50", channels=[dict(kind="affine", g=0.0, scale=spec["disturbance_scale"])],
                     iid_fraction=0.5)
    mine = worlds.population(w, 3, 4)
    out = tempfile.mkdtemp() + "/pop"
    up_pop.run(dict(spec, parents=3, fields=4, save_frames=False), out)
    ref = np.load(out + "/branches.npz")
    print("hand_iid50 err", np.abs(mine["truth"] - ref["truth"]).max())
    assert np.abs(mine["truth"] - ref["truth"]).max() == 0.0
    # snapshot/restore against replay for the composed worlds (forces, goal command, dm_control step counters);
    # 2 parents x 8 draws x 4 actions = 64 branches per parent exceeds the TimeLimit / step-count horizon of a naive restore
    from cwk import cells
    for name in ("E3_fetch_iid50", "E2_cup_iid50", "E1_nu50_gain_iid50"):
        w = cells.STAGE2[name]()
        pop = worlds.population(w, 2, 8)
        env = w.make()
        worst = 0.0
        for i in range(2):
            for j in range(8):
                for a in range(4):
                    env.reset(w.sim_spec("population")["seed"] + i)
                    for t in range(2):
                        env.step(pop["past_actions"][i, t], pop["past_eps"][i, t])
                    r = env.target(env.query(pop["actions"][i, a], pop["eps"][i, j]))
                    worst = max(worst, float(np.abs(r - pop["truth"][i, j, a]).max()))
        env.close()
        print(name, "max |snapshot branch - replay branch| =", worst)
        assert worst < 1e-9
    print("ALL SIM CHECKS PASSED")
