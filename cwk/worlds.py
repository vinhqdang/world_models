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
from cwk.sim import ChannelSimulator
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

    def sim_spec(self, part):
        spec = resolve(registry.simulator(self.base, part))
        spec = copy.deepcopy(spec)
        v = float(self.iid_fraction)
        spec["innovation_variance"] = v
        spec["benchmark_design"] = dict(spec.get("benchmark_design", {}), iid_fraction=v)
        spec["channels"] = copy.deepcopy(self.channels)
        spec["forces"] = copy.deepcopy(self.forces)
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
            for t in range(T):
                loc = env.location(state, t)
                g = np.empty(nd)
                for j in range(nd):
                    innov = (streams[j][0] if t < 2 else streams[j][2]).standard_normal()
                    g[j] = (np.sqrt(1 - v) * env.fields[j](loc, parent)[0] + np.sqrt(v) * innov) if v < 1 else innov
                eps = env.epsilon(g)
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


def population(world, parents, draws, verbose=False, start=0):
    """Counterfactual evaluation population (authors' protocol). Returns arrays as in branches.npz:
    sources (N, 3, D), past_actions (N, 2, p), past_eps (N, 2, nd), actions (N, k, p), eps (N, E, nd),
    truth (N, E, k, D), posterior (N, nd, 2)."""
    spec = world.sim_spec("population")
    env = ChannelSimulator(spec)
    v, nd, k = float(world.iid_fraction), env.n_drivers, K_ACTIONS
    D, p = len(env.target(env.state())), env.action_dim
    n, e = parents, draws
    sources, truth = np.empty((n, 3, D)), np.empty((n, e, k, D))
    past_a, past_eps = np.empty((n, 2, p)), np.empty((n, 2, nd))
    actions, eps, post = np.empty((n, k, p)), np.empty((n, e, nd)), np.empty((n, nd, 2))
    try:
        for ii, i in enumerate(range(start, start + parents)):
            streams = _driver_streams(spec, i, nd)
            prefix, query, noise = streams[0]
            state = env.reset(spec["seed"] + i)
            basis, observed = [], [[] for _ in range(nd)]
            for t in range(2):
                sources[ii, t] = env.target(state)
                loc = env.location(state, t)
                basis.append(env.fields[0].basis(loc))
                g = np.empty(nd)
                for j in range(nd):
                    innov = streams[j][0].standard_normal()
                    g[j] = np.sqrt(1 - v) * env.fields[j](loc, i)[0] + np.sqrt(v) * innov
                    observed[j].append(g[j])
                past_eps[ii, t] = env.epsilon(g)
                past_a[ii, t] = env.behavior_action(t, prefix)
                state = env.step(past_a[ii, t], past_eps[ii, t])
            sources[ii, 2] = env.target(state)
            q_basis = env.fields[0].basis(env.location(state, 2))
            for j in range(nd):
                mu, var = conditional_driver(np.asarray(basis), np.asarray(observed[j]), q_basis, v)
                post[ii, j] = mu, var
                z = streams[j][2].standard_normal(e) if j > 0 else noise.standard_normal(e)
                eps[ii, :, j] = np.tanh(mu + np.sqrt(var) * z)
            actions[ii, 0] = env.fix_actions(query.uniform(-env.action_limit, env.action_limit, p))
            actions[ii, 1:] = env.fix_actions(query.uniform(-env.action_limit, env.action_limit, (k - 1, p)))
            snap = env.snapshot()
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
    return dict(sources=sources, past_actions=past_a, past_eps=past_eps, actions=actions, eps=eps, truth=truth,
                posterior=post, weights=weights, names=list(env.names), action_limit=env.action_limit,
                iid_fraction=v)


def save(path, d):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
