"""Instrumentation for LeWM/stable-worldmodel CEM planning.

Records, for every replanning round of every env:
  * pool statistics of the CEM population (first / last iteration),
  * the model-predicted cost of the executed plan (c_hat),
  * once the plan has been executed, the realised latent cost y
    (distance between the embedding of the *real* resulting observation and the goal embedding).
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch
import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.callbacks import Callback


class PoolRecorder(Callback):
    output_key = 'pool'

    def __init__(self):
        self.history = []
        self.first = None
        self.last = None

    def reset(self):
        self.first = None
        self.last = None

    def start_batch(self):
        pass

    def __call__(self, step, costs, topk_vals, **kw):
        c = costs.detach().float().cpu()
        if step == 0:
            self.first = c
        self.last = c

    def end_solve(self):
        pass


class LoggedCEM(CEMSolver):
    """CEMSolver that exposes the info dict and pool costs of the latest solve."""

    def __init__(self, *a, topk_frac=None, **kw):
        n = kw.get('num_samples', 300)
        if topk_frac is not None:
            kw['topk'] = max(3, int(round(topk_frac * n)))
        super().__init__(*a, **kw)
        self.rec = PoolRecorder()
        self.callbacks.append(self.rec)
        self.last_info = None
        self.last_out = None

    def solve(self, info_dict, init_action=None):
        self.last_info = info_dict
        out = super().solve(info_dict, init_action)
        self.last_out = out
        return out


def _expand(info, S):
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(1).expand(v.shape[0], S, *v.shape[1:])
        elif isinstance(v, np.ndarray):
            out[k] = np.repeat(v[:, None, ...], S, axis=1)
        else:
            out[k] = v
    return out


@torch.inference_mode()
def realised_cost(model, info, device):
    """sum_d (enc(current real obs) - enc(goal))^2 for each env in info (non-expanded)."""
    px = info['pixels'].to(device)
    gl = info['goal'].to(device)
    now = model.encode({'pixels': px[:, -1:]})['emb'][:, -1]
    goal = model.encode({'pixels': gl})['emb'][:, -1]
    return ((now - goal) ** 2).sum(-1).float().cpu()


@torch.inference_mode()
def predicted_cost(model, info, actions, device):
    """Model cost (same criterion as planning) of a single given plan per env."""
    inf = _expand({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in info.items()}, 1)
    plan = actions.to(device=device, dtype=next(model.parameters()).dtype)[:, None]
    return model.get_cost(inf, plan)[:, 0].float().cpu()


class Logger:
    def __init__(self):
        self.rows = []          # one row per executed plan (outcome filled in later)
        self.calls = defaultdict(int)
        self.pending = {}       # env -> row awaiting its realised outcome
        self.meta = {}

    def dump(self, path, extra=None):
        with open(path, 'w') as f:
            json.dump(dict(meta=self.meta, extra=extra or {}, rows=self.rows), f)


LOG = Logger()
LOG.frames = {}
LOG.debug = False
LOG.steps = []
LOG.nstep = 0


def patch_criterion():
    """Make LeWM.criterion correct for batch size > 1 (identical numerics for B == 1)."""
    from stable_worldmodel.wm.lewm.lewm import LeWM

    def criterion(self, info_dict):
        pred = info_dict['predicted_emb']  # (B, S, T, D)
        goal = info_dict['goal_emb']  # (B, T, D) or (B, S, T, D)
        if goal.dim() == 3:
            goal = goal.unsqueeze(1)
        return ((pred[..., -1:, :] - goal[..., -1:, :].detach()) ** 2).sum(dim=(2, 3))

    LeWM.criterion = criterion


def patch_policy(device='cuda'):
    P = swm.policy.WorldModelPolicy
    orig = P.get_action

    def get_action(self, info_dict, **kw):
        n = self.env.num_envs
        model = self.solver.model
        term = info_dict.get('terminated')
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)
        buf_empty = [len(self._action_buffer[i]) == 0 for i in range(n)]
        if LOG.debug:
            try:
                pr = np.asarray(info_dict['proprio'])
                gp = np.asarray(info_dict['goal_proprio'])
                for i in range(min(n, 6)):
                    LOG.steps.append(dict(env=i, t=LOG.nstep, dead=bool(dead[i]), buf=len(self._action_buffer[i]),
                                          pos=pr[i][-1].tolist(), goal=gp[i][-1].tolist()))
                LOG.nstep += 1
            except Exception as e:
                LOG.meta.setdefault('errors', []).append('steps:' + repr(e))
            px = info_dict['pixels']
            for i in range(min(n, 4)):
                if buf_empty[i] or dead[i]:
                    LOG.frames[f'e{i}_call{LOG.calls[i]}_dead{int(dead[i])}'] = np.asarray(px[i][-1]).copy()
                if 'goal' not in LOG.frames and i == 0:
                    pass
                LOG.frames[f'goal{i}'] = np.asarray(info_dict['goal'][i][-1]).copy()
        # 1) realised outcome of the chunk that just finished (or of an episode that just ended)
        try:
            due = [i for i in LOG.pending if buf_empty[i] or dead[i]]
            if due:
                LOG.force_prepare = True
                prepared = self._prepare_info(info_dict)
                LOG.force_prepare = False
                y_all = realised_cost(model, prepared, device)
                pr_now = np.asarray(info_dict['proprio'])
                gp_now = np.asarray(info_dict['goal_proprio'])
                for i in due:
                    row = LOG.pending.pop(i)
                    row['y'] = float(y_all[i])
                    row['p_end'] = float(np.linalg.norm(pr_now[i][-1] - gp_now[i][-1]))
                    row['partial'] = bool(dead[i] and not buf_empty[i])
                    row['dead_at_outcome'] = bool(dead[i])
        except Exception as e:
            LOG.meta.setdefault('errors', []).append('outcome:' + repr(e))
        # 2) planning
        replan = [i for i in range(n) if buf_empty[i] and not dead[i]]
        act = orig(self, info_dict, **kw)
        if not replan:
            return act
        solver = self.solver
        info = solver.last_info
        out = solver.last_out
        try:
            d_start = realised_cost(model, info, device)
            pr_all = np.asarray(info_dict['proprio'])
            gp_all = np.asarray(info_dict['goal_proprio'])
            ch = predicted_cost(model, info, out['actions'], device)
            first, last = solver.rec.first, solver.rec.last
            for r, i in enumerate(replan):
                k = LOG.calls[i]
                LOG.calls[i] += 1
                pl = out['actions'][r]                       # normalised action plan (H, A*block)
                row = dict(env=i, k=k, d_start=float(d_start[r]), c_hat=float(ch[r]),
                           a_absmean=float(pl.abs().mean()), a_absmax=float(pl.abs().max()), a_rms=float(pl.pow(2).mean().sqrt()),
                           p_start=float(np.linalg.norm(pr_all[i][-1] - gp_all[i][-1])))
                if first is not None:
                    f0, l0 = first[r], last[r]
                    row.update(
                        first_min=float(f0.min()), first_mean=float(f0.mean()), first_std=float(f0.std()),
                        last_min=float(l0.min()), last_mean=float(l0.mean()), last_std=float(l0.std()),
                        last_q10=float(l0.quantile(0.1)), last_q50=float(l0.median()),
                    )
                LOG.rows.append(row)
                LOG.pending[i] = row
        except Exception as e:  # never break the evaluation
            LOG.meta.setdefault('errors', []).append('plan:' + repr(e))
        return act

    P.get_action = get_action

    # Speed: the stock policy re-transforms every env's frames at every env step even when no env replans.
    orig_prepare = P._prepare_info

    def fast_prepare(self, info_dict):
        n = self.env.num_envs
        term = info_dict.get('terminated')
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)
        need = any(len(self._action_buffer[i]) == 0 and not dead[i] for i in range(n))
        if not need and not getattr(LOG, 'force_prepare', False):
            return {k: info_dict[k] for k in ('terminated', '_needs_flush') if k in info_dict}
        return orig_prepare(self, info_dict)

    P._prepare_info = fast_prepare


def patch_world(out_path):
    W = swm.World
    orig = W.evaluate

    def evaluate(self, *a, **kw):
        t0 = time.time()
        res = orig(self, *a, **kw)
        LOG.meta['seconds'] = time.time() - t0
        succ = res.get('episode_successes')
        extra = dict(success_rate=res.get('success_rate'),
                     episode_successes=[bool(x) for x in succ] if succ is not None else None)
        extra['steps'] = LOG.steps
        LOG.dump(out_path, extra)
        if LOG.debug:
            np.savez(out_path.replace('.json', '_frames.npz'), **LOG.frames)
        return res

    W.evaluate = evaluate
