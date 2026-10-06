"""S2 Phase B closed-loop evaluation on cliff_hi with learned es models: base planner, CVaR mix, failure-rate controller, SCC wrapper.

python experiments/s2_eval.py --ckpt suite_hi/ckpt/es_s0.pt --m 0 --seed 2000 --arm scc --alpha 0.10 --calfit results/s2/calfit_N32.npz --J 4 --out results/s2/runs/x.json

Protocol (fixed-J, slot-wise common random numbers): E=16 slots; the j-th episode of slot i has the same start/goal AND the same wind-sign
sequence (by episode step) in every arm run with the same --seed and --phase (selwm.s2_scc.SlotEnv). Each slot runs exactly J episodes
(j0..j0+J-1), so every arm has E*J episodes per run, no streaming-window length bias, and exact (slot, j) pairing. Pairing is partial because
planner randomness and trajectories differ. Slots that finished their J episodes are dropped from the planning batch.
Shift conditions: --burn B runs B steps of streaming episodes at wind 0.09 (phase 0, state of the controller / ACI carried over), then all slots restart
(phase 1) at wind --wind and run J episodes. Arms without state (base, cvar, static SCC) need no burn-in.

Arms: base (CEM, mean score) | cvar (CVaR mix --lam) | ctrl (failure-rate controller, target --control) | scc | pi0 | pi0fast (no model).
SCC (all quantities on the fall-probability component of the planner cost, see results/s2/REPORT.md):
  stage 1: executed plan = mean of the final elites; pf = model fall probability of that plan (8 chained-CRN particles); U1 = pf + q1;  commit if U1 <= b
  stage 2 (escalate): candidates = executed plan + top-3 members; re-score with --Mp fresh independent particles; best = argmin total score;
           U2 = pf64(best) + q2;  commit best if U2 <= b
  otherwise DEFER: pi_0 action.
  q = split-conformal quantile (level 1-alpha) of residuals pT - pf of the executed plan (band 'sel') or of random final-population members / candidates ('pw').
  --aci 1: alpha_t update  alpha_{t+1} = alpha_t + gamma (alpha - err_t),  err_t = 1[F_t > U_t] for committed steps once their outcome F_t
           of a counterfactual replay is observed: F_t = 1 if the committed plan, executed OPEN LOOP for H steps on the wind realised during the next H
           closed-loop steps (read off the observed transitions, selwm.s2_scc.ReplayLog), enters the pit. F_t is an unbiased single-sample of the true
           open-loop fall probability pT of the committed plan. q_t = quantile of the stored calibration residuals at level 1-alpha_t.
           (The realised closed-loop fall indicator is also stored, as Fcl; its rate is far below alpha, so an update based on it would never bind.)
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm import s2_scc as S
from selwm.noise_aware import NoiseAwareScorer, RiskController

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=None); ap.add_argument('--m', type=int, default=0)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--arm', required=True, choices=['base', 'cvar', 'ctrl', 'scc', 'rand', 'pi0', 'pi0fast'])
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--lam', type=float, default=0.5); ap.add_argument('--control', type=float, default=0.05); ap.add_argument('--eta', type=float, default=0.05)
ap.add_argument('--alpha', type=float, default=0.10); ap.add_argument('--band', default='sel', choices=['sel', 'pw'])
ap.add_argument('--pdefer', type=float, default=0.15, help='rand arm: probability of deferring to pi_0 at a step, independent of the state')
ap.add_argument('--aci', type=int, default=0); ap.add_argument('--gamma', type=float, default=0.02)
ap.add_argument('--b', type=float, default=None); ap.add_argument('--calfit', default=None); ap.add_argument('--Mp', type=int, default=64)
ap.add_argument('--J', type=int, default=4); ap.add_argument('--j0', type=int, default=0); ap.add_argument('--E', type=int, default=16)
ap.add_argument('--wind', type=float, default=0.09); ap.add_argument('--burn', type=int, default=0)
ap.add_argument('--Mt', type=int, default=100, help='true-simulator particles for the DIAGNOSTIC true fall probability of the committed plan')
ap.add_argument('--diag', type=int, default=1); ap.add_argument('--out', required=True)
args = ap.parse_args()
FORBIDDEN = {0, 1, 2, 100, 101, 102, 200, 201, 202, 211, 300, 301, 302, 400, 401, 402, 403, 404, 405, 500, 501, 502, 600, 601, 602, 650, 651, 652, 700, 701, 702, 900}
assert args.seed >= 1000 and args.seed not in FORBIDDEN
np.random.seed(args.seed); torch.manual_seed(args.seed)
E = args.E
need_model = args.arm not in ('pi0', 'pi0fast')
if need_model:
    mem, anchor, lm, lm_ind = S.load_models(args.ckpt)
    cnt = S.count_rows(mem)
gen = torch.Generator().manual_seed(args.seed); gen_t = torch.Generator().manual_seed(args.seed + 500_000)

# ----------------------------------------------------------------------------------------------- SCC gate
class Gate:
    def __init__(self, calfit, m, alpha, band, aci, gamma, b):
        d = np.load(calfit)
        tag = 'a%02d' % int(round(alpha * 100))
        pre = '' if band == 'sel' else 'p'
        self.R1 = np.sort(d[f'{pre}R1_m{m}']); self.R2 = np.sort(d[f'{pre}R2_{tag}_m{m}'])
        self.b = float(d['b']) if b is None else b
        self.alpha, self.alpha_t, self.aci, self.gamma = alpha, alpha, aci, gamma
        self.trace = []
        self.n_upd = 0; self.n_err = 0

    def q(self):
        lvl = 1 - (self.alpha_t if self.aci else self.alpha)
        return S.level_quantile(self.R1, lvl), S.level_quantile(self.R2, lvl)

    def update(self, F, U):
        err = 1.0 if (F > U) else 0.0
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err), 0.002, 0.5))
        self.n_upd += 1; self.n_err += int(err)

gate = Gate(args.calfit, args.m, args.alpha, args.band, args.aci, args.gamma, args.b) if args.arm == 'scc' else None
ctrl = RiskController(args.control, args.eta, lam0=0.3) if args.arm == 'ctrl' else None
scorer0 = NoiseAwareScorer(args.lam if args.arm == 'cvar' else 0.0, 0.1, False)

rng_rand = np.random.default_rng([5151, args.seed]); n_rand_defer = [0]
episodes = []; STEP = {k: [] for k in ['tag', 'slot', 'jep', 'tep', 'stage', 'pf1', 'U1', 'pf2', 'U2', 'pT1', 'pT2', 'Fcl', 'Frp', 'alpha_t', 'q1', 'q2']}
RL = S.ReplayLog(E, args.seed)
n_env_steps = 0; pend_idx = {}


def run_phase(env, tag, T=None, J=None, j0=0):
    global n_env_steps
    active = np.ones(E, bool); prev_reset = np.ones(E, bool)
    for i in range(E): RL.reset(i)
    if need_model: lm.noise.state = None
    tracker = S.OutcomeTracker(E)
    t = 0
    while active.any() and (T is None or t < T):
        idx = np.where(active)[0]; n = len(idx)
        pos = env.p[idx].copy(); goal = env.goal[idx].copy()
        rec = None
        if args.arm == 'pi0':
            act = S.pi0_action(pos, goal, S.PI0)
        elif args.arm == 'pi0fast':
            act = S.pi0_action(pos, goal, S.PI0_FAST)
        else:
            p = torch.tensor(pos, dtype=torch.float32); fell = torch.tensor(env.fell[idx])
            z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(goal, dtype=torch.float32))
            lm.begin_replan(reset_mask=prev_reset[idx])
            sc = NoiseAwareScorer(ctrl.lam, 0.1, False) if ctrl else scorer0
            pop = S.cem_pop(lm, z0, zg, args.N, args.M, args.iters, gen, sc)
            plan = pop['plan']
            act = plan[:, 0].numpy().copy()
            if args.arm == 'rand':
                dm = rng_rand.random(n) < args.pdefer
                if dm.any(): act[dm] = S.pi0_action(pos[dm], goal[dm], S.PI0)
                n_rand_defer[0] += int(dm.sum())
            if args.arm == 'scc':
                q1, q2 = gate.q()
                j, pf = S.score_exec(lm, z0, zg, plan, args.M, gen)
                pf = pf.numpy(); U1 = pf + q1
                stage = np.where(U1 <= gate.b, 1, 0)
                pf2 = np.full(n, np.nan); U2 = np.full(n, np.nan); U_used = U1.copy()
                best_plan = plan.clone()
                esc = np.where(stage == 0)[0]
                if len(esc):
                    cand = S.candidates(pop)[esc]
                    jc, pfc = S.rescore(lm_ind, z0[esc], zg[esc], cand, args.Mp, gen)
                    bi = jc.argmin(1); ar = torch.arange(len(esc))
                    pfb = pfc[ar, bi].numpy(); Ub = pfb + q2
                    bp = cand[ar, bi]
                    pf2[esc] = pfb; U2[esc] = Ub
                    ok = Ub <= gate.b
                    for e_i, k in enumerate(esc):
                        best_plan[k] = bp[e_i]
                        if ok[e_i]:
                            act[k] = bp[e_i, 0].numpy(); stage[k] = 2; U_used[k] = Ub[e_i]
                    dk = esc[~ok]
                    if len(dk): act[dk] = S.pi0_action(pos[dk], goal[dk], S.PI0)
                pT1 = pT2 = np.full(n, np.nan)
                if args.diag:
                    both = torch.stack([plan, best_plan], 1)
                    _, pT = S.true_cost_w(lm, mem, anchor, p, zg, both, args.Mt, gen_t, env.wind)
                    pT1 = pT[:, 0].numpy(); pT2 = pT[:, 1].numpy()
                rec = dict(stage=stage, pf1=pf, U1=U1, pf2=pf2, U2=U2, pT1=pT1, pT2=pT2, U=U_used, q1=q1, q2=q2,
                           plan_c=np.where((stage == 2)[:, None, None], best_plan.numpy(), plan.numpy()))
        n_env_steps += n
        tep = env.t_ep[idx].copy(); yb = env.p[idx, 1].copy()
        env.step(act, idx)
        for ii, i in enumerate(idx): RL.log(i, yb[ii], env.p[i, 1], act[ii, 1])
        fin = env.finished(idx)
        base_k = len(STEP['tag'])
        if rec is not None:
            for ii, i in enumerate(idx):
                for key in ('stage', 'pf1', 'U1', 'pf2', 'U2', 'pT1', 'pT2'): STEP[key].append(float(rec[key][ii]))
                STEP['tag'].append(tag); STEP['slot'].append(int(i)); STEP['jep'].append(int(env.j[i])); STEP['tep'].append(int(tep[ii]))
                STEP['Fcl'].append(-1.0); STEP['Frp'].append(-1.0); STEP['alpha_t'].append(gate.alpha_t); STEP['q1'].append(rec['q1']); STEP['q2'].append(rec['q2'])
                tracker.add(i, (base_k + ii, int(rec['stage'][ii]), float(rec['U'][ii]), rec['plan_c'][ii], pos[ii].copy(), int(tep[ii])))
        prev_reset[:] = False
        for ii, i in enumerate(idx):
            if rec is not None:
                for (k, st, U, plan_c, p0_, t0_), F in tracker.after_step(i, bool(env.fell[i]), bool(fin[ii])):
                    Frp = RL.replay(i, plan_c, p0_, t0_) if st >= 1 else -1
                    STEP['Fcl'][k] = float(F); STEP['Frp'][k] = float(Frp)
                    if st >= 1 and gate.aci: gate.update(Frp, U)
                    elif st >= 1: gate.n_upd += 1; gate.n_err += int(Frp > U)
            if fin[ii]:
                o = env.outcome(i)
                episodes.append([int(i), int(env.j[i]), o, int(env.t_ep[i]), tag])
                if ctrl: ctrl.update([o == 'fall'])
                if J is not None and env.j[i] >= j0 + J - 1:
                    active[i] = False
                else:
                    env.next_episode(i); prev_reset[i] = True; RL.reset(i)
        t += 1
        if t % 50 == 0:
            print(f'[{tag}] t={t} active={int(active.sum())} eps={len(episodes)} {time.time()-t0:.0f}s' + (f' alpha_t={gate.alpha_t:.3f}' if gate else '') + (f' lam={ctrl.lam:.2f}' if ctrl else ''), file=sys.stderr, flush=True)
    return t


t0 = time.time()
def make_env(phase, wind, j0):
    env = S.SlotEnv(args.seed, E, wind, phase)
    env.j[:] = j0
    for i in range(E): env._new_episode(i)
    return env

steps = {}
if args.burn > 0:
    steps['pre'] = run_phase(make_env(0, 0.09, 0), 'pre', T=args.burn)
steps['post'] = run_phase(make_env(1, args.wind, args.j0), 'post', J=args.J, j0=args.j0)
ep = [e for e in episodes if e[4] == 'post']
n = max(1, len(ep)); cn = {o: sum(1 for e in ep if e[2] == o) for o in ('success', 'fall', 'timeout')}
sg = [e[3] for e in ep if e[2] == 'success']
res = dict(args=vars(args), episodes_n=len(ep), success=cn['success'] / n, fall=cn['fall'] / n, timeout=cn['timeout'] / n, median_steps=float(np.median(sg)) if sg else None,
           episodes=episodes, steps=steps, rows_per_env_step=(cnt['rows'] / max(1, n_env_steps) if need_model else 0.0), env_steps=n_env_steps, sec=time.time() - t0)
if gate:
    res['gate'] = dict(b=gate.b, alpha_final=gate.alpha_t, n_upd=gate.n_upd, n_err=gate.n_err, q1_n=len(gate.R1), q2_n=len(gate.R2))
if args.arm == 'rand': res['defer_rate'] = n_rand_defer[0] / max(1, n_env_steps)
if ctrl:
    res['lam_final'] = ctrl.lam; res['lam_mean'] = float(np.mean(ctrl.hist)) if ctrl.hist else None
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
json.dump(res, open(args.out, 'w'))
if STEP['tag']: np.savez_compressed(args.out.replace('.json', '.steps.npz'), **{k: (np.array(v) if k != 'tag' else np.array(v)) for k, v in STEP.items()})
print(json.dumps({k: v for k, v in res.items() if k not in ('args', 'episodes')}))
