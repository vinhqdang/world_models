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
from selwm.jepa import JEPA, LatentModel, EnsembleLatent, FailureAnchor, observe
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
    env.step(plan[:, 0].cpu().numpy()); t_ep += 1
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
           lam_final=float(log_lam[-1]), lam_mean=float(np.mean(log_lam)), sec=time.time() - t0)
print(json.dumps({k: v for k, v in res.items() if k != 'args'}))
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
