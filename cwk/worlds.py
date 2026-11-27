"""World specifications, factual collection and counterfactual populations.

A world is a configuration of the authors' simulators (experiments/configs/sims of the upstream repository) with the
disturbance expressed as channels (``cwk.sim``). Populations follow the authors' protocol (cwm/sims/population.py,
MIT licence): parent i replays two behaviour-policy transitions, then k = 4 actions (the executed one and three
alternatives, uniform on the action support) are queried under each of E draws of the scored transition's disturbance,
drawn from the driver law given the prefix. Every branch starts from an exact snapshot of the simulator after the prefix
(verified against replay in experiments/cwk_test_sim.py).
"""
import copy
import json
from dataclasses import dataclass, field

import numpy as np

from cwk import upstream  # noqa: F401
from cwk.sim import ChannelSimulator, law_eps
from cwm import registry
from cwm.sims import resolve
from cwm.sims.fields import conditional_driver

K_ACTIONS = 4


@dataclass
class World:
    name: str
    base: str                              # upstream config name
    channels: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    iid_fraction: float = 1.0
    overrides: dict = field(default_factory=dict)
    note: str = ""
    seed_offset: int = 0
    laws: object = None                    # per-driver law (dict for all drivers or a list); None: tanh (Stage 0-2 cells)

    def sim_spec(self, part):
        spec = resolve(registry.simulator(self.base, part))
        spec = copy.deepcopy(spec)
        v = float(self.iid_fraction)
        spec["innovation_variance"] = v
        spec["benchmark_design"] = dict(spec.get("benchmark_design", {}), iid_fraction=v)
        spec["channels"] = copy.deepcopy(self.channels)
        spec["forces"] = copy.deepcopy(self.forces)
        if self.laws is not None:
            spec["driver_laws"] = copy.deepcopy(self.laws)
        spec.update(copy.deepcopy(self.overrides))
        spec["seed"] = int(spec["seed"]) + self.seed_offset
        spec.pop("save_frames", None)
        return spec

    def make(self, part="population"):
        return ChannelSimulator(self.sim_spec(part))


def groups(world):
    d = registry.simulator(world.base, "population")
    return resolve(d)["benchmark_design"]["coordinate_groups"], resolve(d)["benchmark_design"]["primary_group"]


def _driver_streams(spec, parent, n_drivers):
    """Per-driver innovation generators; driver 0 uses the authors' streams, others independent ones."""
    seed = spec["seed"]
    rngs = [[np.random.default_rng(np.random.SeedSequence([seed, parent, tag])) for tag in (317, 337, 357)]]
    extra = [[np.random.default_rng(np.random.SeedSequence([seed, parent, 4000 + 10 * j + c])) for c in range(3)]
             for j in range(1, n_drivers)]
    return rngs + extra


def gate_stream(seed, parent, driver, kind, n):
    """Uniform stream behind the gate of ``driver``: one independent deterministic stream per (seed, parent, driver) and
    kind (query draws / prefix transitions); the entry index is the draw (or transition). A draw's gate is therefore a
    function of (seed, parent, draw, driver) only and is shared by every counterfactual branch of that draw."""
    tag = {"query": 5000, "prefix": 6000}[kind] + int(driver)
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(parent), tag])).random(n)


def gates_for(env, seed, parent, kind, n):
    """(n, nd) 0/1 gates (all ones for drivers whose law is not gated); None when the world has no per-driver laws."""
    if env.laws is None:
        return None
    out = np.ones((n, env.n_drivers))
    for j, law in enumerate(env.laws):
        if law.get("kind") == "gated":
            out[:, j] = gate_stream(seed, parent, j, kind, n) < law["pi"]
    return out


def collect(world, n, T=5, start=0, verbose=False):
    """Factual episodes of T transitions: states (n, T+1, D), actions (n, T, p), eps (n, T, nd), drivers.
    Transitions 0, 1: behaviour policy; later transitions: uniform actions on the support."""
    spec = world.sim_spec("collection")
    env = ChannelSimulator(spec)
    v, nd = float(world.iid_fraction), env.n_drivers
    D, p = len(env.target(env.state())), env.action_dim
    S, A, E = np.empty((n, T + 1, D)), np.empty((n, T, p)), np.empty((n, T, nd))
    try:
        for e, parent in enumerate(range(start, start + n)):
            streams = _driver_streams(spec, parent, nd)
            prefix = streams[0][0]
            query = streams[0][1]
            state = env.reset(spec["seed"] + parent)
            S[e, 0] = env.target(state)
            gts = gates_for(env, spec["seed"], parent, "prefix", T)
            for t in range(T):
                loc = env.location(state, t)
                g = np.empty(nd)
                for j in range(nd):
                    innov = (streams[j][0] if t < 2 else streams[j][2]).standard_normal()
                    g[j] = (np.sqrt(1 - v) * env.fields[j](loc, parent)[0] + np.sqrt(v) * innov) if v < 1 else innov
                eps = env.epsilon(g) if gts is None else env.epsilon(g, gts[t])
                action = (env.behavior_action(t, prefix) if t < 2 else
                          env.fix_actions(query.uniform(-env.action_limit, env.action_limit, p)))
                state = env.step(action, eps)
                A[e, t], E[e, t], S[e, t + 1] = action, eps, env.target(state)
            if verbose and (e + 1) % 1000 == 0:
                print("collect", e + 1, "/", n, flush=True)
    finally:
        env.close()
    return dict(states=S, actions=A, eps=E, iid_fraction=v, names=list(env.names),
                action_limit=env.action_limit)


def population(world, parents, draws, verbose=False, start=0, fd_h=None):
    """Counterfactual evaluation population (authors' protocol). Returns arrays as in branches.npz:
    sources (N, 3, D), past_actions (N, 2, p), past_eps (N, 2, nd), actions (N, k, p), eps (N, E, nd),
    truth (N, E, k, D), posterior (N, nd, 2). With ``fd_h`` also the structural linearisation of every parent:
    fd_y0 (N, k, D) outcome at eps = 0 and fd_J (N, k, D, nd) central finite-difference Jacobian in the drivers."""
    spec = world.sim_spec("population")
    env = ChannelSimulator(spec)
    v, nd, k = float(world.iid_fraction), env.n_drivers, K_ACTIONS
    D, p = len(env.target(env.state())), env.action_dim
    n, e = parents, draws
    sources, truth = np.empty((n, 3, D)), np.empty((n, e, k, D))
    past_a, past_eps = np.empty((n, 2, p)), np.empty((n, 2, nd))
    actions, eps, post = np.empty((n, k, p)), np.empty((n, e, nd)), np.empty((n, nd, 2))
    fy0, fJ = np.empty((n, k, D)), np.empty((n, k, D, nd))
    try:
        for ii, i in enumerate(range(start, start + parents)):
            streams = _driver_streams(spec, i, nd)
            prefix, query, noise = streams[0]
            state = env.reset(spec["seed"] + i)
            basis, observed = [], [[] for _ in range(nd)]
            gp = gates_for(env, spec["seed"], i, "prefix", 2)
            gq = gates_for(env, spec["seed"], i, "query", e)
            for t in range(2):
                sources[ii, t] = env.target(state)
                loc = env.location(state, t)
                basis.append(env.fields[0].basis(loc))
                g = np.empty(nd)
                for j in range(nd):
                    innov = streams[j][0].standard_normal()
                    g[j] = np.sqrt(1 - v) * env.fields[j](loc, i)[0] + np.sqrt(v) * innov
                    observed[j].append(g[j])
                past_eps[ii, t] = env.epsilon(g) if gp is None else env.epsilon(g, gp[t])
                past_a[ii, t] = env.behavior_action(t, prefix)
                state = env.step(past_a[ii, t], past_eps[ii, t])
            sources[ii, 2] = env.target(state)
            q_basis = env.fields[0].basis(env.location(state, 2))
            for j in range(nd):
                mu, var = conditional_driver(np.asarray(basis), np.asarray(observed[j]), q_basis, v)
                post[ii, j] = mu, var
                z = streams[j][2].standard_normal(e) if j > 0 else noise.standard_normal(e)
                if env.laws is None:
                    eps[ii, :, j] = np.tanh(mu + np.sqrt(var) * z)
                else:
                    eps[ii, :, j] = law_eps(env.laws[j], mu + np.sqrt(var) * z, gq[:, j])
            actions[ii, 0] = env.fix_actions(query.uniform(-env.action_limit, env.action_limit, p))
            actions[ii, 1:] = env.fix_actions(query.uniform(-env.action_limit, env.action_limit, (k - 1, p)))
            snap = env.snapshot()
            if fd_h is not None:
                for a in range(k):
                    env.restore(snap)
                    fy0[ii, a] = env.target(env.query(actions[ii, a], np.zeros(nd)))
                    for j in range(nd):
                        ys = []
                        for sgn in (1.0, -1.0):
                            env.restore(snap)
                            ev = np.zeros(nd)
                            ev[j] = sgn * fd_h
                            ys.append(env.target(env.query(actions[ii, a], ev)))
                        fJ[ii, a, :, j] = (ys[0] - ys[1]) / (2 * fd_h)
            for jj in range(e):
                for a in range(k):
                    env.restore(snap)
                    truth[ii, jj, a] = env.target(env.query(actions[ii, a], eps[ii, jj]))
            if verbose and (ii + 1) % 32 == 0:
                print("parents", ii + 1, "/", n, flush=True)
    finally:
        env.close()
    rng = np.random.default_rng(spec["seed"] + 317)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=512)
    out = dict(sources=sources, past_actions=past_a, past_eps=past_eps, actions=actions, eps=eps, truth=truth,
               posterior=post, weights=weights, names=list(env.names), action_limit=env.action_limit,
               iid_fraction=v)
    if fd_h is not None:
        out.update(fd_y0=fy0, fd_J=fJ, fd_h=fd_h)
    return out


def save(path, d):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
