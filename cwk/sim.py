"""Multi-driver simulator on top of the authors' NativeSimulator (cwm.sims.native, MIT licence).

The environments and their physics are used unchanged. The disturbance is a vector of independent drivers
eps_j = tanh(driver_j), each held for a transition, entering through declared channels:

    kind "affine"    command channel   a + s eps d [(1 - g) + g x],           x = a / (action_gain action_limit)
                     (g = 0 offset, g = 1 pure gain; with one channel this is exactly the authors' actuator_bias /
                     actuator_affine)
    kind "gain"      a + s eps d m(x) with a profile m: "linear" (m = x), "sat" (tanh(c x) / tanh c, misspecified),
                     "deadzone" (zero for |x| < z, linear after, misspecified), "offset" (m = 1)
    kind "rotate"    a + s eps J x on two axes (thrust misalignment, J = [[0, -1], [1, 0]]); x = a / (action_gain action_limit)
    forces           world-frame force eps F on a body during the transition (the authors' body_force, composable)

A command channel may carry ``axis`` (index into the declared axes: the channel acts on that actuator only, direction
(-1)^axis, i.e. a gain fault on one actuator) or ``dir`` (explicit direction over the declared axes). Without either, the
direction is the authors' alternating +-1 over the declared axes (old cells are unchanged).

Per-driver laws (``driver_laws``, optional): eps_j = tanh(g) (default), gated: Bern(pi) (2 Phi(g) - 1) with a gate taken
from an independent deterministic stream, gauss_clip: clip(g / c, -1, 1).

``position_goal_bias`` (Fetch) is applied after the command channels, as in the authors' code.
"""
import copy

import numpy as np
from scipy.special import ndtr

from cwk import upstream  # noqa: F401
from cwm.sims.native import NativeSimulator


def profile(kind, x, **kw):
    if kind in ("offset",):
        return np.ones_like(x)
    if kind in ("linear", "gain"):
        return x
    if kind == "sat":
        c = kw.get("c", 3.0)
        return np.tanh(c * x) / np.tanh(c)
    if kind == "deadzone":
        z = kw.get("z", 0.4)
        return np.sign(x) * np.maximum(np.abs(x) - z, 0.0) / (1 - z)
    if kind == "affine":
        g = kw.get("g", 1.0)
        return (1 - g) + g * x
    raise ValueError(kind)


def law_eps(law, g, gate=None):
    """Driver value eps in [-1, 1] from the Gaussian driver ``g`` and (gated law) a 0/1 gate; broadcasting."""
    kind = (law or {}).get("kind", "tanh")
    g = np.asarray(g, dtype=np.float64)
    if kind == "tanh":
        return np.tanh(g)
    if kind == "gated":
        return np.asarray(gate, dtype=np.float64) * (2.0 * ndtr(g) - 1.0)
    if kind == "gauss_clip":
        return np.clip(g / law.get("c", 3.0), -1.0, 1.0)
    raise ValueError(kind)


class ChannelSimulator(NativeSimulator):
    def __init__(self, spec):
        spec = copy.deepcopy(spec)
        self.channels = spec.pop("channels", [])
        self.forces = spec.pop("forces", [])
        laws = spec.pop("driver_laws", None)
        self.goal_command = spec.get("mechanism") == "position_goal_bias"
        spec["mechanism"] = "actuator_bias"      # parent: no plant set-up; the plant is handled here
        spec.pop("gain_fraction", None)
        super().__init__(spec)
        self.n_drivers = len(self.channels) + len(self.forces)
        self.axes = list(spec.get("axes", range(self.action_dim)))
        if laws is None:
            self.laws = None
        else:
            laws = [laws] * self.n_drivers if isinstance(laws, dict) else list(laws)
            assert len(laws) == self.n_drivers, "one law per driver"
            self.laws = laws
        self.reference = self.action_gain * self.action_limit
        self.fields = []
        from cwm.sims.fields import native_field
        for j in range(max(self.n_drivers, 1)):
            self.fields.append(native_field(dict(spec, seed=int(spec["seed"]) + 1000003 * j), len(self.state())))
        self.field = self.fields[0]
        self._force_ids = []
        if self.forces:
            import mujoco
            if self.family == "dmc":
                self._m, self._d = self.env.physics.model.ptr, self.env.physics.data.ptr
            else:
                self._m, self._d = self.env.model, self.env.data
            for f in self.forces:
                self._force_ids.append(mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, f["body"]))

    # ---- disturbance -------------------------------------------------------------------------------------------
    def epsilon(self, driver, gates=None):
        driver = np.atleast_1d(np.asarray(driver, dtype=np.float64))
        if self.laws is None:
            return np.tanh(driver)
        gates = np.ones(len(driver)) if gates is None else np.asarray(gates, dtype=np.float64)
        return np.array([float(law_eps(self.laws[j], driver[j], gates[j])) for j in range(len(driver))])

    def channel_direction(self, ch):
        if "dir" in ch:
            v = np.zeros(self.action_dim)
            v[self.axes] = ch["dir"]
            return v
        if "axis" in ch:
            v = np.zeros(self.action_dim)
            v[self.axes[ch["axis"]]] = (-1.0) ** ch["axis"]
            return v
        return self.direction

    def command_perturbation(self, action, eps):
        """Normalised-command perturbation of the command channels for the (already gain-scaled) ``action``."""
        x = np.asarray(action, dtype=np.float64) / self.reference
        out = np.zeros_like(x)
        for j, ch in enumerate(self.channels):
            kind = ch.get("kind", "affine")
            kw = {k: v for k, v in ch.items() if k not in ("kind", "scale", "dir", "axis")}
            if kind == "rotate":
                a0, a1 = self.axes[0], self.axes[1]
                out[a0] = out[a0] - ch["scale"] * float(eps[j]) * x[a1]
                out[a1] = out[a1] + ch["scale"] * float(eps[j]) * x[a0]
                continue
            if kind == "affine":
                m = profile("affine", x, g=ch.get("g", 1.0))
            else:
                m = profile(kind, x, **kw)
            out = out + ch["scale"] * float(eps[j]) * self.channel_direction(ch) * m
        return out

    # ---- snapshot / restore (exact: the full MjData is copied) ----------------------------------------------------
    def _mj(self):
        if self.family == "dmc":
            return self.env.physics.model.ptr, self.env.physics.data.ptr
        return self.env.model, self.env.data

    def snapshot(self):
        import mujoco
        m, d = self._mj()
        snap = mujoco.MjData(m)
        mujoco.mj_copyData(snap, m, d)
        return snap, copy.deepcopy(self.last_ob), self.done, self._counters()

    def _counters(self):
        """Python-side step counters (gymnasium TimeLimit, dm_control step count) that MjData does not hold."""
        out = {}
        for key, obj in (("tl", getattr(self, "episode_env", None)), ("dm", self.env if self.family == "dmc" else None)):
            if obj is None:
                continue
            for attr in ("_elapsed_steps", "_step_count", "_reset_next_step"):
                if hasattr(obj, attr):
                    out[(key, attr)] = getattr(obj, attr)
        return out

    def restore(self, snap):
        import mujoco
        m, d = self._mj()
        mujoco.mj_copyData(d, m, snap[0])
        self.last_ob, self.done = copy.deepcopy(snap[1]), snap[2]
        for (key, attr), val in snap[3].items():
            obj = self.episode_env if key == "tl" else self.env
            setattr(obj, attr, val)

    # ---- one native step -------------------------------------------------------------------------------------------
    def _native_step(self, action, epsilon):
        if self.done:
            return self.state()
        action = np.asarray(action)
        eps = np.atleast_1d(np.asarray(epsilon, dtype=np.float64))
        normalized = action + self.command_perturbation(action, eps)
        if self.goal_command:
            center, radius, current = self.workspace()
            width = len(current)
            normalized = normalized.copy()
            normalized[:width] = .98 * (center + radius * normalized[:width] - current) / (2 * radius)
        if np.max(np.abs(normalized)) > 1 + 1e-12:
            raise ValueError("the disturbed command left the native action range")
        actual = (self.low + self.high) / 2 + normalized * (self.high - self.low) / 2
        env = self.env if self.family in {"dmc", "softgym"} else self.episode_env
        nc = len(self.channels)
        if self.forces:
            for j, f in enumerate(self.forces):
                self._d.xfrc_applied[self._force_ids[j], :3] += f["scale"] * float(eps[nc + j]) * np.asarray(
                    f["vector"], dtype=float)
        try:
            result = env.step(actual)
        finally:
            for i in self._force_ids:
                self._d.xfrc_applied[i, :3] = 0.0
        if self.family == "dmc":
            self.last_ob, self.done = result.observation, result.last()
        else:
            self.last_ob = result[0]
            self.done = bool(result[2] or result[3]) if len(result) == 5 else bool(result[2])
        return self.state()
