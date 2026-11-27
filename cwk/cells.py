"""Named worlds: Stage 0/1 (authors' HandReach dial cells) and Stage 2 (rank >= 2 cells)."""
import numpy as np

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
    cell = Cell(world.name, col, pop, g["coords"], g["groups"], list(axes), spec.get("action_limit", 0.35),
                spec["action_gain"], nominal=nominal, true_cmd=cmd, dev=dev, iid=world.iid_fraction,
                primary=primary)
    cell.world = world
    cell.r = len(world.channels) + len(world.forces)
    return cell


# ----------------------------------------------------------------------------------------------------------------
# Paper cells (analysis paper, paper_cf/PLAN.md section 3). Amplitudes are calibrated on simulator-only development
# cohorts by experiments/paper_calibrate.py (keys "info_*", "misalign_*", "force_unit_*" in CALIBRATION.json; the Stage 2
# keys are untouched). Physics untouched; mechanisms composed by cwk/sim.py.

INFO_ENV = {   # environment tag -> (upstream config, family, action limit, base seed offset)
    "E1o": dict(base="hand_affine100", fam="hand", alim=0.45, off=8_000_000),
    "E2o": dict(base="cup", fam="cup", alim=0.35, off=8_100_000),
    "E3o": dict(base="fetch", fam="fetch", alim=0.48, off=8_200_000),
}
OFFSET_SHARE = 0.5          # budget share of the common-offset channel on every axis (r >= d + 1)
GAUSS_CLIP_C = 3.0          # gauss_clip: eps = clip(driver / 3, -1, 1)
PI_DEFAULT = 0.2
VAR_LAW = {"tanh": 0.394, "gauss": 0.111, "gated": None}      # Var(eps) for the variance-matched starting amplitude


def law_spec(law, pi=PI_DEFAULT):
    if law == "tanh":
        return dict(kind="tanh")
    if law == "gated":
        return dict(kind="gated", pi=float(pi))
    if law in ("gauss", "gauss_clip"):
        return dict(kind="gauss_clip", c=GAUSS_CLIP_C)
    raise ValueError(law)


def info_key(env, law, pi, r):
    lw = {"tanh": "tanh", "gauss": "gauss", "gauss_clip": "gauss"}.get(law) or f"gated_p{int(round(pi * 100))}"
    return f"info_{env}_{lw}_r{r}"


def info_dim(env):
    return {"E1o": 2, "E2o": 2, "E3o": 3}[env]


def force_spec(env, scale):
    if env == "E1o":
        from cwm.sims import resolve
        sp = resolve(registry.simulator("hand_load", "population"))
        return dict(body=sp["force_body"], vector=sp["force_vector"], scale=scale)
    if env == "E2o":
        return dict(body="ball", vector=[1.0, 0.0, 0.0], scale=scale)
    return dict(body="robot0:gripper_link", vector=[1.0, 0.0, 0.0], scale=scale)


def info_channels(env, r, sigma):
    """Drivers of an information-limited cell: r = d gains (one per actuator), + common offset (r = d + 1),
    + a force on a body (r = d + 2). Per-axis command budget sigma (no clipping: g alim + sigma <= 0.98 + 0.02 sigma)."""
    d = info_dim(env)
    off = OFFSET_SHARE if r >= d + 1 else 0.0
    ch = [dict(kind="gain", axis=j, scale=sigma * (1 - off)) for j in range(d)]
    if r >= d + 1:
        ch.append(dict(kind="offset", scale=sigma * off))
    return ch


def info_cell(env, law="gated", pi=PI_DEFAULT, r=None, iid=1.0, tag=None, sigma=None, calkey=None, seed_offset=None):
    """Information-limited cell INFOLIM(env, law, pi, r). ``sigma``: per-axis command budget (default: calibrated, else
    a variance-matched starting value); the force scale is force_unit * sigma."""
    e = INFO_ENV[env]
    d = info_dim(env)
    r = d + 1 if r is None else int(r)
    assert d <= r <= d + 2
    key = calkey or info_key(env, law, pi, r)
    c = _cal(key, None)
    if sigma is not None:
        c = dict(sigma=sigma, gain=0.98 * (1 - sigma) / e["alim"])
    elif c is None:
        var = (pi / 3.0) if law == "gated" else VAR_LAW["tanh" if law == "tanh" else "gauss"]
        sg = float(min(0.9, 0.7 * np.sqrt(0.394 / var)))
        c = dict(sigma=sg, gain=0.98 * (1 - sg) / e["alim"])
    ch = info_channels(env, r, c["sigma"])
    forces = []
    if r == d + 2:
        fu = _cal(f"force_unit_{env}", dict(unit=1.0))["unit"]
        forces = [force_spec(env, fu * c["sigma"])]
    lw = law_spec(law, pi)
    name = tag or f"{env}_{law}{'' if law != 'gated' else int(round(pi * 100))}_r{r}_iid{int(iid * 100)}"
    return World(name, e["base"], channels=ch, forces=forces, iid_fraction=iid, overrides=dict(action_gain=c["gain"]),
                 seed_offset=e["off"] + (0 if seed_offset is None else seed_offset), laws=lw,
                 note=f"INFOLIM {env} law={law} pi={pi} r={r} sigma={c['sigma']}")


def misalign_cell(env="E2o", law="tanh", offset=True, iid=1.0, tag=None, sigma=None, pi=PI_DEFAULT, seed_offset=None):
    """MISALIGN: thrust-misalignment driver (command perturbation s eps J x on the two actuated axes) plus, optionally, an
    independent common-offset driver. Ball-in-Cup by default (d = 2)."""
    e = INFO_ENV[env]
    key = f"misalign_{env}_{law}{'' if offset else '_nooff'}"
    c = _cal(key, None)
    if sigma is not None:
        c = dict(sigma=sigma, gain=0.98 * (1 - sigma) / e["alim"])
    elif c is None:
        c = dict(sigma=0.6, gain=0.98 * 0.4 / e["alim"])
    ch = [dict(kind="rotate", scale=c["sigma"] * (OFFSET_SHARE if offset else 1.0))]
    if offset:
        ch.append(dict(kind="offset", scale=c["sigma"] * OFFSET_SHARE))
    name = tag or f"MIS_{env}_{law}{'' if offset else '_nooff'}_iid{int(iid * 100)}"
    return World(name, e["base"], channels=ch, iid_fraction=iid, overrides=dict(action_gain=c["gain"]),
                 seed_offset=e["off"] + 500_000 + (0 if seed_offset is None else seed_offset), laws=law_spec(law, pi),
                 note=f"MISALIGN {env} rotate{' + offset' if offset else ''}")


def _upstream_cell(name, base, iid=1.0):
    """The authors' own configuration (offset fault, one driver), for the no-harm check."""
    if base == "hand_load":
        return load_cell(name, "hand_load", iid)
    g = {"hand_affine50": 0.5, "hand_affine100": 1.0}.get(base, 0.0)
    return dial_cell(name, base, g, iid)


UPSTREAM = {
    "hand": lambda: dial_cell("hand_l0", "hand", 0.0),
    "cup": lambda: dial_cell("cup", "cup", 0.0),
    "fetch": lambda: dial_cell("fetch", "fetch", 0.0),
    "hand_iid50": lambda: dial_cell("hand_iid50", "hand_iid50", 0.0, iid=0.5),
    "cup_iid50": lambda: dial_cell("cup_iid50", "cup_iid50", 0.0, iid=0.5),
    "fetch_iid50": lambda: dial_cell("fetch_iid50", "fetch_iid50", 0.0, iid=0.5),
    "hand_load": lambda: load_cell("hand_load"),
    "hand_affine50": STAGE0["hand_affine50"],
    "hand_affine100": STAGE0["hand_affine100"],
}

# the information-limited design (R2a): name -> constructor
INFOLIM = {}
for _env in INFO_ENV:
    _d = info_dim(_env)
    INFOLIM[f"{_env}_P"] = (lambda env=_env: info_cell(env, "gated", 0.2, None, 1.0, tag=f"{env}_P"))
    INFOLIM[f"{_env}_T"] = (lambda env=_env: info_cell(env, "tanh", None, None, 1.0, tag=f"{env}_T"))
    INFOLIM[f"{_env}_G"] = (lambda env=_env: info_cell(env, "gauss", None, None, 1.0, tag=f"{env}_G"))
    for _pi in (0.05, 0.1, 0.5, 1.0):
        INFOLIM[f"{_env}_pi{int(_pi * 100)}"] = (lambda env=_env, pi=_pi: info_cell(env, "gated", pi, None, 1.0, tag=f"{env}_pi{int(pi * 100)}"))
    INFOLIM[f"{_env}_rd"] = (lambda env=_env: info_cell(env, "gated", 0.2, info_dim(env), 1.0, tag=f"{env}_rd"))
    INFOLIM[f"{_env}_rd2"] = (lambda env=_env: info_cell(env, "gated", 0.2, info_dim(env) + 2, 1.0, tag=f"{env}_rd2"))
    INFOLIM[f"{_env}_Piid50"] = (lambda env=_env: info_cell(env, "gated", 0.2, None, 0.5, tag=f"{env}_Piid50", calkey=info_key(env, "gated", 0.2, info_dim(env) + 1) + "_iid50"))
    INFOLIM[f"{_env}_Pdata1"] = (lambda env=_env: info_cell(env, "gated", 0.2, None, 1.0, tag=f"{env}_Pdata1", seed_offset=1_000_000, calkey=info_key(env, "gated", 0.2, info_dim(env) + 1)))
    INFOLIM[f"{_env}_Pdata2"] = (lambda env=_env: info_cell(env, "gated", 0.2, None, 1.0, tag=f"{env}_Pdata2", seed_offset=2_000_000, calkey=info_key(env, "gated", 0.2, info_dim(env) + 1)))
MISALIGN = {"MIS_E2o": lambda: misalign_cell("E2o", "tanh", True),
            "MIS_E2o_nooff": lambda: misalign_cell("E2o", "tanh", False)}
ALL_PAPER = {**{k: v for k, v in INFOLIM.items()}, **MISALIGN, **{f"UP_{k}": v for k, v in UPSTREAM.items()}}


SEALED_JSON = "/home/user/world_models/results/paper/sealed_cell.json"


def sealed_cell(path=SEALED_JSON):
    """The sealed cell: environment, pi and amplitude drawn by experiments/paper_sealed.py from an RNG seeded with the protocol
    commit hash (written to ``path``)."""
    c = json.load(open(path))
    return info_cell(c["env"], "gated", c["pi"], None, 1.0, tag="SEALED", sigma=c["sigma"], seed_offset=c["seed_offset"])
