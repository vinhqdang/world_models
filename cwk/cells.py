"""Named worlds: Stage 0/1 (authors' HandReach dial cells) and Stage 2 (rank >= 2 cells)."""
from cwk import upstream  # noqa: F401
from cwk.worlds import World
from cwm import registry
from cwm.sims import resolve


def _scale(base):
    return resolve(registry.simulator(base, "population"))["disturbance_scale"]


def dial_cell(name, base, g, iid=1.0):
    """The authors' single-driver cell: one channel a + s eps d [(1 - g) + g x] (g = 0: offset; 1: pure gain)."""
    return World(name, base, channels=[dict(kind="affine", g=g, scale=_scale(base))], iid_fraction=iid,
                 note="authors' configuration, one driver")


def load_cell(name, base="hand_load", iid=1.0):
    spec = resolve(registry.simulator(base, "population"))
    return World(name, base, forces=[dict(body=spec["force_body"], vector=spec["force_vector"],
                                          scale=spec["disturbance_scale"])], iid_fraction=iid)


STAGE0 = {
    "hand_l0": lambda: dial_cell("hand_l0", "hand", 0.0),
    "hand_affine50": lambda: dial_cell("hand_affine50", "hand_affine50", 0.5),
    "hand_affine100": lambda: dial_cell("hand_affine100", "hand_affine100", 1.0),
    "hand_load": lambda: load_cell("hand_load"),
}


# ----------------------------------------------------------------------------------------------------------------
# Stage 2: rank >= 2 cells. Amplitudes (sigma) and the command gain are calibrated on simulator-only development
# cohorts to the authors' design target (outcome-blind oracle R^2 about 50 +- 8 on the primary group), see
# experiments/cwk_calibrate.py and results/cwk/CALIBRATION.json.
import json
import os

_CAL = "/home/user/world_models/results/cwk/CALIBRATION.json"


def _cal(key, default):
    try:
        return json.load(open(_CAL))[key]
    except Exception:
        return default


def e1_cell(nu, profile="gain", iid=0.5, tag=None, sigma=None, seed_offset=7_000_000):
    """HandReach actuator_affine with two independent drivers: offset (weight 1 - nu) and gain (weight nu, profile
    gain / sat / deadzone). Total amplitude sigma, so that the no-clipping bound g alim + sigma <= 0.98 holds."""
    key = f"e1_nu{int(nu * 100)}_{profile}"
    c = _cal(key, None)
    if sigma is not None:
        c = dict(sigma=sigma, gain=0.98 * (1 - sigma) / 0.45)
    elif c is None:                                # paper's equilibrium calibration (App. D) as the starting point
        m = (1 - nu) ** 2 + nu ** 2 / 3
        sig = 1 / (1 + 3 * 0.394 * m / 0.98)
        c = dict(sigma=sig, gain=0.98 * (1 - sig) / 0.45)
    ch = [dict(kind="offset", scale=c["sigma"] * (1 - nu)), dict(kind=profile, scale=c["sigma"] * nu)]
    name = tag or f"E1_nu{int(nu * 100)}_{profile}_iid{int(iid * 100)}"
    return World(name, "hand_affine100", channels=ch, iid_fraction=iid, overrides=dict(action_gain=c["gain"]),
                 seed_offset=seed_offset, note=f"HandReach offset+{profile} drivers, gain weight nu={nu}")


def e2_cell(iid=0.5, tag=None, sigma=None, wind=None, seed_offset=7_100_000):
    """Ball-in-Cup: gain driver on the actuators plus an independent wind force on the ball."""
    c = _cal("e2", dict(sigma=0.45, gain=0.9, wind=0.6))
    if sigma is not None:
        c = dict(sigma=sigma, gain=0.98 * (1 - sigma) / 0.35, wind=wind)
    name = tag or f"E2_cup_iid{int(iid * 100)}"
    return World(name, "cup", channels=[dict(kind="gain", scale=c["sigma"])],
                 forces=[dict(body="ball", vector=[1.0, 0.0, 0.0], scale=c["wind"])], iid_fraction=iid,
                 overrides=dict(action_gain=c["gain"]), seed_offset=seed_offset, note="cup gain + wind on ball")


def e3_cell(iid=0.5, tag=None, sigma=None, load=None, seed_offset=7_200_000):
    """FetchReach: gain driver on the goal command plus a load force on the gripper."""
    c = _cal("e3", dict(sigma=0.45, gain=0.9, load=30.0))
    if sigma is not None:
        c = dict(sigma=sigma, gain=0.98 * (1 - sigma) / 0.48, load=load)
    name = tag or f"E3_fetch_iid{int(iid * 100)}"
    return World(name, "fetch", channels=[dict(kind="gain", scale=c["sigma"])],
                 forces=[dict(body="robot0:gripper_link", vector=[1.0, 0.0, 0.0], scale=c["load"])],
                 iid_fraction=iid, overrides=dict(action_gain=c["gain"]), seed_offset=seed_offset,
                 note="fetch gain + load on gripper")


STAGE2 = {}
for _iid in (0.5, 1.0):
    for _nu in (0.5, 0.8):
        _n = f"E1_nu{int(_nu * 100)}_gain_iid{int(_iid * 100)}"
        STAGE2[_n] = (lambda nu=_nu, iid=_iid: e1_cell(nu, "gain", iid))
    STAGE2[f"E2_cup_iid{int(_iid * 100)}"] = (lambda iid=_iid: e2_cell(iid))
    STAGE2[f"E3_fetch_iid{int(_iid * 100)}"] = (lambda iid=_iid: e3_cell(iid))
for _p in ("sat", "deadzone"):
    for _iid in (0.5, 1.0):
        STAGE2[f"E1_nu50_{_p}_iid{int(_iid * 100)}"] = (lambda p=_p, iid=_iid: e1_cell(0.5, p, iid))


# Kernel / transport coordinates = the scored primary group of each environment (d = 2, 2, 3). A first version of the
# pipeline modelled all 24 HandReach joint angles; the neural polarization head under-fitted there within the budget
# (held-out NLL 39.6 against 22.1 for the flow head, wrist R^2 0.73 against 0.96 for the closed-form moment estimator on
# lambda = 1 seed 0), so every arm now works on the scored coordinates only (documented deviation, see STAGE2.md).
GROUPS = {
    "hand": dict(coords=[0, 1], groups={"wrist": [0, 1]}, primary="wrist"),
    "cup": dict(coords=[0, 1], groups={"cup_position": [0, 1]}, primary="cup_position"),
    # the two finger coordinates are fixed by the declared fixed action (zero variance): not scored
    "fetch": dict(coords=[0, 1, 2], groups={"gripper_position": [0, 1, 2]}, primary="gripper_position"),
}


def family(world):
    return "hand" if world.base.startswith("hand") else ("cup" if world.base == "cup" else "fetch")


def pipeline_cell(world, col, pop, dev=None):
    """cwk.pipeline.Cell for a world (structure knowledge: actuated axes, nominal single-driver fault, true channels)."""
    from cwk.pipeline import Cell
    spec = world.sim_spec("population")
    g = GROUPS[family(world)]
    cmd = [c for c in world.channels]
    tot = sum(c["scale"] for c in cmd)
    gain_w = sum(c["scale"] for c in cmd if c.get("kind") in ("gain", "sat", "deadzone", "affine")) / max(tot, 1e-9)
    # nominal single-driver fault type: eps (offset part + gain part): kind affine with g = gain share, amplitude = total
    gshare = sum(c["scale"] * (c.get("g", 1.0) if c.get("kind") == "affine" else (1.0 if c.get("kind") in ("gain", "sat", "deadzone") else 0.0)) for c in cmd) / max(tot, 1e-9)
    nominal = [dict(kind="affine", g=float(gshare), scale=float(tot))]
    axes = spec.get("axes", list(range(col["actions"].shape[-1])))
    primary = g["primary"]
    return Cell(world.name, col, pop, g["coords"], g["groups"], list(axes), spec.get("action_limit", 0.35),
                spec["action_gain"], nominal=nominal, true_cmd=cmd, dev=dev, iid=world.iid_fraction,
                primary=primary)
