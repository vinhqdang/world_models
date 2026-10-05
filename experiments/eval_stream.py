"""Streaming closed-loop evaluation (episodes restart on finish), optional wind shift and failure-rate control.

Modes
  --planner mean            deterministic rollout of the predictor, risk-neutral on the mean embedding
  --planner risk            sample particles; score (1-lam)*E + lam*CVaR_0.1;  --lam fixed or --control delta
Options  --shrink 1 (noise-aware EB selection)  --race 1 (successive halving)  --crn 1
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, LatentModel, EnsembleLatent, FailureAnchor, observe, es_scale_loss
from selwm.risk_plan import cem
from selwm.noise_aware import NoiseAwareScorer, cem_raced, RiskController

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True, help='checkpoint, or comma-separated checkpoints for --planner ensemble')
ap.add_argument('--planner', default='risk'); ap.add_argument('--stage_w', type=float, default=1.0)
ap.add_argument('--failcost', type=int, default=1); ap.add_argument('--kappa', type=float, default=3.0)
ap.add_argument('--variant', default='cliff'); ap.add_argument('--E', type=int, default=64)
ap.add_argument('--N', type=int, default=128); ap.add_argument('--M', type=int, default=16)
ap.add_argument('--H', type=int, default=12); ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--total_steps', type=int, default=400); ap.add_argument('--ep_len', type=int, default=120)
ap.add_argument('--lam', type=float, default=0.0); ap.add_argument('--tail', type=float, default=0.1)
ap.add_argument('--shrink', type=int, default=0); ap.add_argument('--race', type=int, default=0); ap.add_argument('--crn', type=int, default=0)
ap.add_argument('--control', type=float, default=-1, help='target failure rate delta (<0: off)'); ap.add_argument('--eta', type=float, default=0.05)
ap.add_argument('--tta_gate', type=float, default=0.0, help='z-threshold of the drift gate (0 = always adapt)'); ap.add_argument('--tta_replay', type=float, default=0.0)
ap.add_argument('--tta_params', default='all'); ap.add_argument('--tta_batch', type=int, default=256); ap.add_argument('--tta_window', type=int, default=25)
ap.add_argument('--tta', default='', help='online predictor fine-tuning with the model-native loss: mse|nll|es'); ap.add_argument('--tta_lr', type=float, default=3e-4); ap.add_argument('--tta_steps', type=int, default=2)
ap.add_argument('--adapt', type=int, default=0); ap.add_argument('--per_dim', type=int, default=0); ap.add_argument('--oracle_vec', default=''); ap.add_argument('--adapt_lr', type=float, default=0.03); ap.add_argument('--oracle_scale', type=float, default=0.0)
ap.add_argument('--shift_at', type=int, default=-1); ap.add_argument('--shift_wind', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default=None)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
dev = torch.device(args.device)

def load(path):
    ck = torch.load(path, map_location=dev); a = ck['args']
    m = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')).to(dev); m.load_state_dict(ck['state']); m.eval()
    return m

def start_goal_scale(model_enc, mode):
    e = StochNav(args.variant, 1, 4242)
    sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = model_enc(observe(torch.tensor(sp, dtype=torch.float32, device=dev), None, args.variant, mode))
        zg_ = model_enc(observe(torch.tensor(gp, dtype=torch.float32, device=dev), None, args.variant, mode))
    return float(((zs - zg_) ** 2).sum(-1).median())

if args.planner == 'ensemble':
    members = [load(c) for c in args.ckpt.split(',')]
    anchors = [FailureAnchor(m.encode, args.variant, m.obs, device=dev) for m in members] if args.failcost else None
    scale = float(np.mean([start_goal_scale(m.encode, m.obs) for m in members]))
    lm = EnsembleLatent(members, args.variant, stage_w=args.stage_w, anchors=anchors, kappa=args.kappa, scale=scale)
else:
    mem = load(args.ckpt)
    anchor = FailureAnchor(mem.encode, args.variant, mem.obs, device=dev) if args.failcost else None
    lm = LatentModel(mem, args.variant, stochastic=(args.planner != 'mean'), crn=bool(args.crn), stage_w=args.stage_w,
                     anchor=anchor, kappa=args.kappa, scale=start_goal_scale(mem.encode, mem.obs))
log_s = None
if args.planner == 'risk' and (args.adapt or args.oracle_scale > 0 or args.oracle_vec):
    D = lm.m.pred.dim
    if args.oracle_vec:
        log_s = torch.tensor(np.log([float(x) for x in args.oracle_vec.split(',')]), dtype=torch.float32, requires_grad=False)
    elif args.per_dim:
        log_s = torch.zeros(D, requires_grad=bool(args.adapt))
    else:
        log_s = torch.tensor(float(np.log(args.oracle_scale)) if args.oracle_scale > 0 else 0.0, requires_grad=bool(args.adapt))
    lm.log_s = log_s
    s_opt = torch.optim.Adam([log_s], lr=args.adapt_lr) if args.adapt else None
s_hist = []
tta_opt, tta_buf = None, []
if args.tta:
    base_model = lm.m if args.planner != 'ensemble' else None
    for p_ in base_model.pred.parameters(): p_.requires_grad_(True)
    params = list(base_model.pred.parameters()) if args.tta_params == 'all' else list(base_model.pred.out.parameters())
    for p_ in base_model.pred.parameters(): p_.requires_grad_(False)
    for p_ in params: p_.requires_grad_(True)
    tta_opt = torch.optim.Adam(params, lr=args.tta_lr)
    import glob as _g
    src = None
    cand = sorted(_g.glob(f'runs/data_{args.variant}_300000_e0.25.npz'))
    if cand and (args.tta_replay > 0 or args.tta_gate > 0):
        _d = np.load(cand[0]); n_ = len(_d['P'])
        sel = np.random.default_rng(0).choice(n_, 60000, replace=False)
        src = [torch.tensor(_d[k][sel], dtype=torch.float32 if k in ('P', 'A', 'P2') else torch.bool, device=dev) for k in ['P', 'F', 'A', 'P2', 'F2']]
    def src_batch(n):
        i = torch.randint(0, len(src[0]), (n,), device=dev)
        return (observe(src[0][i], src[1][i], args.variant, base_model.obs), src[2][i], observe(src[3][i], src[4][i], args.variant, base_model.obs))
    gate_mu = gate_sd = None; win = []; gate_on = args.tta_gate <= 0
    if args.tta_gate > 0:
        with torch.no_grad():
            ls_ = []
            for _ in range(200):
                b = src_batch(16); ls_.append(float(base_model.loss(b[0], b[1], b[2], M=8)[0]))
        gate_mu, gate_sd = float(np.mean(ls_)), float(np.std(ls_))
    gate_log = []
ctrl = RiskController(args.control, args.eta, lam0=0.3) if args.control >= 0 else None

env = StochNav(args.variant, args.E, 5000 + args.seed); env.reset()
gen = torch.Generator(device=dev).manual_seed(args.seed)
t_ep = np.zeros(args.E, int)
stats = dict(done=0, fall=0, success=0, timeout=0, steps_to_goal=[])
phase_stats = {}
log_lam = []
t0 = time.time()
M_stages, keep = [4, 4, 8], [0.25, 0.125]
for t in range(args.total_steps):
    if t == args.shift_at:
        env.wind_base = args.shift_wind
    phase = 'after' if (args.shift_at >= 0 and t >= args.shift_at) else 'before'
    p = torch.tensor(env.p, dtype=torch.float32, device=dev); fell = torch.tensor(env.fell, device=dev)
    z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32, device=dev))
    lam = ctrl.lam if ctrl else args.lam
    if args.planner == 'mean':
        plan = cem(lm, z0, zg, args.H, args.N, 1, iters=args.iters, risk='expected', gen=gen)
    elif args.race:
        plan = cem_raced(lm, z0, zg, args.H, args.N, M_stages, keep, iters=args.iters, lam=lam, tail=args.tail, shrink=bool(args.shrink), gen=gen)
    else:
        plan = cem(lm, z0, zg, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=NoiseAwareScorer(lam, args.tail, bool(args.shrink)))
    alive_before = ~(env.fell | env.reached)
    p_before, f_before = env.p.copy(), env.fell.copy()
    act0 = plan[:, 0].cpu().numpy()
    env.step(act0); t_ep += 1
    if args.adapt and log_s is not None and alive_before.any():
        with torch.no_grad():
            z1 = lm.obs_to_latent(torch.tensor(env.p, dtype=torch.float32, device=dev), torch.tensor(env.fell, device=dev))
        ab = torch.tensor(alive_before, device=dev)
        loss = es_scale_loss(lm, z0[ab], torch.tensor(act0, dtype=torch.float32, device=dev)[ab], z1[ab], M=8)
        s_opt.zero_grad(); loss.backward(); s_opt.step()
        with torch.no_grad(): log_s.clamp_(-1.0, 2.0)
    if tta_opt is not None and alive_before.any():
        o0 = observe(torch.tensor(p_before, dtype=torch.float32, device=dev)[alive_before], torch.tensor(f_before, device=dev)[alive_before], args.variant, base_model.obs)
        o1 = observe(torch.tensor(env.p, dtype=torch.float32, device=dev)[alive_before], torch.tensor(env.fell, device=dev)[alive_before], args.variant, base_model.obs)
        ac = torch.tensor(act0, dtype=torch.float32, device=dev)[alive_before]
        tta_buf.append((o0, ac, o1)); tta_buf[:] = tta_buf[-200:]
        if args.tta_gate > 0:
            with torch.no_grad():
                win.append(float(base_model.loss(o0[:16], ac[:16], o1[:16], M=8)[0])); win[:] = win[-args.tta_window:]
            if len(win) >= args.tta_window:
                z = (np.mean(win) - gate_mu) / (gate_sd / np.sqrt(len(win)))
                gate_on = z > args.tta_gate if not gate_on else z > 0.5 * args.tta_gate
                gate_log.append((t, float(z), bool(gate_on)))
        if gate_on:
            for _ in range(args.tta_steps):
                idx = np.random.randint(0, len(tta_buf), 8)
                O0 = torch.cat([tta_buf[i][0] for i in idx]); AC = torch.cat([tta_buf[i][1] for i in idx]); O1 = torch.cat([tta_buf[i][2] for i in idx])
                if src is not None and args.tta_replay > 0:
                    n_src = int(args.tta_replay * args.tta_batch)
                    b = src_batch(n_src); O0 = torch.cat([O0, b[0]]); AC = torch.cat([AC, b[1]]); O1 = torch.cat([O1, b[2]])
                loss, _ = base_model.loss(O0, AC, O1, M=8)
                tta_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); tta_opt.step()
    s_hist.append(float(log_s.exp().detach().mean()) if log_s is not None else 1.0)
    if log_s is not None and log_s.numel() > 1: s_vec = log_s.exp().detach().tolist()
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    if done.any():
        failed = env.fell[done]
        ps = phase_stats.setdefault(phase, dict(done=0, fall=0, success=0, timeout=0))
        for i in np.where(done)[0]:
            if not env.fell[i] and not env.reached[i]: stats.setdefault('to_pos', []).append(env.p[i].tolist()); stats.setdefault('to_goal', []).append(env.goal[i].tolist())
            ps['done'] += 1; stats['done'] += 1
            if env.fell[i]: ps['fall'] += 1; stats['fall'] += 1
            elif env.reached[i]: ps['success'] += 1; stats['success'] += 1; stats['steps_to_goal'].append(int(t_ep[i]))
            else: ps['timeout'] += 1; stats['timeout'] += 1
        if ctrl: ctrl.update(failed)
        s, g = env.sample_start_goal(int(done.sum()))
        env.p[done] = s; env.goal[done] = g; env.fell[done] = False; env.reached[done] = False; t_ep[done] = 0
    log_lam.append(lam)
n = max(1, stats['done'])
res = dict(args=vars(args), episodes=stats['done'], success=stats['success'] / n, fall=stats['fall'] / n, timeout=stats['timeout'] / n,
           median_steps=float(np.median(stats['steps_to_goal'])) if stats['steps_to_goal'] else None,
           phases={k: dict(v, success_rate=v['success'] / max(1, v['done']), fall_rate=v['fall'] / max(1, v['done'])) for k, v in phase_stats.items()},
           timeout_final_dist=(float(np.mean(np.linalg.norm(np.array(stats['to_pos']) - np.array(stats['to_goal']), axis=1))) if stats.get('to_pos') else None),
           timeout_mean_xy=(np.mean(stats['to_pos'], 0).round(3).tolist() if stats.get('to_pos') else None),
           spread_scale_final=s_hist[-1], spread_vec_final=(s_vec if 's_vec' in globals() else None), spread_scale_mean_before=float(np.mean(s_hist[:max(1, args.shift_at)])) if s_hist else 1.0,
           spread_scale_mean_after=float(np.mean(s_hist[args.shift_at:])) if (args.shift_at >= 0 and len(s_hist) > args.shift_at) else None,
           gate_first_on=(next((g[0] for g in gate_log if g[2]), None) if (args.tta and args.tta_gate > 0) else None),
           gate_frac_on=(float(np.mean([g[2] for g in gate_log])) if (args.tta and args.tta_gate > 0 and gate_log) else None),
           lam_final=float(log_lam[-1]), lam_mean=float(np.mean(log_lam)), sec=time.time() - t0)
print(json.dumps({k: v for k, v in res.items() if k != 'args'}))
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
