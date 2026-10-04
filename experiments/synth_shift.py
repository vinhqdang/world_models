"""Where fixed tuning cannot work: (S2) mid-stream error shift, (S3) heterogeneous contexts.

Policies (all see only the TRUE cost of the single iterate they execute):
  last        default CEM (final iterate)
  fixed_val   best fixed iterate tuned on the first 300 rounds (an oracle-free-at-deploy 'validation' protocol)
  fixed_hind  best fixed iterate in hindsight over the whole stream (not attainable)
  on_global   ours, one optimism curve, all history
  on_window   ours, one optimism curve, sliding window W
  on_ctx      ours, per-context curves (context id given), sliding window
"""
import sys, json, math, torch, numpy as np
sys.path.insert(0, '.')
from selwm.synth import SynthWorld
from selwm.pressure import cem_trajectory, OptimismCurve
torch.set_num_threads(4)
H, A, N, ITERS = 5, 4, 200, 12


class World2(SynthWorld):
    def Jhat_ctx(self, a, a_star, s1):                      # s1: (B,)
        amp = self.s0 + s1[:, None] * (a ** 2).sum(-1) / self.D
        return self.J(a, a_star) + amp * self.field(a)


def rounds(world, s1_vec, seed):
    R = len(s1_vec)
    gen = torch.Generator().manual_seed(seed)
    a_star = world.new_contexts(R, gen)
    s1 = torch.tensor(s1_vec, dtype=torch.float32)
    mus, kls = cem_trajectory(lambda acts: world.Jhat_ctx(acts.flatten(2), a_star, s1), R, H, A, N, ITERS, gen=gen, init_std=1.0)
    mu_f = mus.flatten(2)
    chat = torch.stack([world.Jhat_ctx(m[:, None], a_star, s1)[:, 0] for m in mu_f]).numpy()
    J = torch.stack([world.J(m[:, None], a_star)[:, 0] for m in mu_f]).numpy()
    return kls.numpy(), chat, J


def run_online(kls, chat, J, ctx, rng, window=None, per_ctx=False, warm=20, eps_floor=0.05):
    K1, R = kls.shape
    grid = np.arange(1, K1)
    hist = []                                                 # (t, ctx, kl, e)
    chosen = np.zeros(R, int)
    for t in range(R):
        cur = OptimismCurve()
        for (tt, cc, kl, e) in hist:
            if (window is None or t - tt <= window) and (not per_ctx or cc == ctx[t]):
                cur.add(kl, e)
        cur.fit()
        eps = 1.0 if t < warm else max(eps_floor, 1.0 / math.sqrt(t + 1))
        if per_ctx and len(cur.kl) < 15:
            eps = 1.0                                         # explore until this context has data
        if rng.random() < eps:
            k = rng.choice(grid)
        else:
            k = grid[np.argmin(chat[grid, t] + cur(kls[grid, t]))]
        chosen[t] = k
        hist.append((t, ctx[t], kls[k, t], J[k, t] - chat[k, t]))
    return chosen


def evaluate(name, s1_vec, ctx, phases, seeds=range(8)):
    res = {}
    for seed in seeds:
        w = World2(s1=0.5, ell=1.0, seed=seed)
        kls, chat, J = rounds(w, s1_vec, seed=200 + seed)
        R = J.shape[1]
        rng = np.random.default_rng(seed)
        pol = {
            'last': np.full(R, ITERS),
            'on_global': run_online(kls, chat, J, ctx, rng),
            'on_window': run_online(kls, chat, J, ctx, rng, window=300),
            'on_ctx': run_online(kls, chat, J, ctx, rng, window=600, per_ctx=True),
        }
        kv = J[1:, :300].mean(1).argmin() + 1
        pol['fixed_val'] = np.full(R, kv)
        kh = J[1:].mean(1).argmin() + 1
        pol['fixed_hind'] = np.full(R, kh)
        for p, ch in pol.items():
            Jp = J[ch, np.arange(R)]
            for ph, sl in phases.items():
                res.setdefault((p, ph), []).append(float(Jp[sl].mean()))
        res.setdefault(('oracle_r', 'all'), []).append(float(J[1:].min(0).mean()))
    return res


def table(title, res, phases):
    print('\n' + title)
    pols = ['last', 'fixed_val', 'fixed_hind', 'on_global', 'on_window', 'on_ctx']
    print(f'{"policy":11s}' + ''.join(f'{ph:>20s}' for ph in phases))
    for p in pols:
        print(f'{p:11s}' + ''.join(f'{np.mean(res[(p, ph)]):12.4f} +/-{np.std(res[(p, ph)], ddof=1)/math.sqrt(len(res[(p, ph)])):.4f}' for ph in phases))
    print(f'{"oracle_r":11s}  (per-round best, all rounds) {np.mean(res[("oracle_r","all")]):.4f}')


R = 1600
# S2: error amplitude jumps 4x at t = R/2
s1 = np.where(np.arange(R) < R // 2, 0.15, 0.6)
ph2 = {'before shift': slice(300, R // 2), 'after shift': slice(R // 2 + 300, R)}
r2 = evaluate('S2', s1, np.zeros(R, int), ph2)
table('S2: mid-stream shift (s1: 0.15 -> 0.6), true cost of executed plans', r2, ph2)
# S3: two interleaved context types with different reliability
rng = np.random.default_rng(5)
ctx = rng.integers(0, 2, R)
s1c = np.where(ctx == 0, 0.08, 0.7)
ph3 = {'ctx0 (reliable)': None, 'ctx1 (unreliable)': None}
res3 = {}
for seed in range(8):
    w = World2(s1=0.5, ell=1.0, seed=seed)
    kls, chat, J = rounds(w, s1c, seed=300 + seed)
    rr = np.random.default_rng(seed)
    pol = {'last': np.full(R, ITERS), 'on_global': run_online(kls, chat, J, ctx, rr),
           'on_window': run_online(kls, chat, J, ctx, rr, window=300),
           'on_ctx': run_online(kls, chat, J, ctx, rr, window=600, per_ctx=True)}
    pol['fixed_val'] = np.full(R, J[1:, :300].mean(1).argmin() + 1)
    pol['fixed_hind'] = np.full(R, J[1:].mean(1).argmin() + 1)
    late = np.arange(R) >= 300
    for p, ch in pol.items():
        Jp = J[ch, np.arange(R)]
        for c, nm in [(0, 'ctx0 (reliable)'), (1, 'ctx1 (unreliable)')]:
            res3.setdefault((p, nm), []).append(float(Jp[late & (ctx == c)].mean()))
    res3.setdefault(('oracle_r', 'all'), []).append(float(J[1:].min(0)[late].mean()))
table('S3: heterogeneous contexts (s1 = 0.08 vs 0.7), true cost of executed plans', res3, list(ph3))
json.dump({'S2': {f'{k[0]}|{k[1]}': v for k, v in r2.items()}, 'S3': {f'{k[0]}|{k[1]}': v for k, v in res3.items()}},
          open('results/synth_shift.json', 'w'), indent=1)
