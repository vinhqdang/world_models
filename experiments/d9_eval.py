"""D9: feedback-parameterised CEM x coupled (common random numbers) noise, streaming closed-loop evaluation on the
learned energy-score latent model (derived from d3_eval.py / d6_eval.py).
  --fb none|const|step   open-loop CEM, or CEM over (nominal actions, deviation-feedback gain) with kmode const|step
  --scheme indep|crn|crn_chain3   noise scheme (selwm.d6_model.SCHEMES)
Per-episode outcomes are stored so that bootstrap CIs can be computed over episodes."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.risk_plan import cem
from selwm.d3_fb import fb_cem
from selwm.d6_model import make_noise
from selwm.d9_model import CoupledLatentFB
from selwm.noise_aware import NoiseAwareScorer

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--fb', default='none', choices=['none', 'const', 'step'])
ap.add_argument('--scheme', default='indep')
ap.add_argument('--kmax', type=float, default=7.5)
ap.add_argument('--stage_w', type=float, default=1.0); ap.add_argument('--failcost', type=int, default=1); ap.add_argument('--kappa', type=float, default=3.0)
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--E', type=int, default=16)
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--H', type=int, default=10); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--total_steps', type=int, default=450); ap.add_argument('--ep_len', type=int, default=120)
ap.add_argument('--seed', type=int, default=211); ap.add_argument('--out', default=None)
args = ap.parse_args()
dev = torch.device('cpu')
np.random.seed(args.seed); torch.manual_seed(args.seed)

ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')).to(dev)
mem.load_state_dict(ck['state']); mem.eval()
assert mem.kind == 'es'

def start_goal_scale(enc, mode):
    e = StochNav(args.variant, 1, 4242)
    sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = enc(observe(torch.tensor(sp, dtype=torch.float32, device=dev), None, args.variant, mode))
        zg_ = enc(observe(torch.tensor(gp, dtype=torch.float32, device=dev), None, args.variant, mode))
    return float(((zs - zg_) ** 2).sum(-1).median())

anchor = FailureAnchor(mem.encode, args.variant, mem.obs, device=dev) if args.failcost else None
lm = CoupledLatentFB(mem, args.variant, noise=make_noise(args.scheme), stochastic=True, stage_w=args.stage_w, anchor=anchor,
                     kappa=args.kappa, scale=start_goal_scale(mem.encode, mem.obs))

# count single-state predictor evaluations (compute accounting)
cnt = dict(calls=0, rows=0)
_orig = mem.pred.forward
def _counted(z, *r, **k):
    cnt['calls'] += 1; cnt['rows'] += int(z.numel() // z.shape[-1])
    return _orig(z, *r, **k)
mem.pred.forward = _counted

env = StochNav(args.variant, args.E, 5000 + args.seed); env.reset()
gen = torch.Generator(device=dev).manual_seed(args.seed)
t_ep = np.zeros(args.E, int)
prev_reset = np.zeros(args.E, bool)
episodes = []        # (outcome, steps)
K_hist = []
t0 = time.time()
scorer = NoiseAwareScorer(0.0, 0.1, False)
for t in range(args.total_steps):
    p = torch.tensor(env.p, dtype=torch.float32, device=dev); fell = torch.tensor(env.fell, device=dev)
    z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32, device=dev))
    lm.begin_replan(reset_mask=prev_reset)
    if args.fb == 'none':
        plan = cem(lm, z0, zg, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=scorer)
    else:
        plan, Kc = fb_cem(lm, z0, zg, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=scorer, kmode=args.fb, kmax=args.kmax)
        K_hist.append(Kc.mean((0, 1)).tolist())
    env.step(plan[:, 0].cpu().numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    if done.any():
        for i in np.where(done)[0]:
            o = 'fall' if env.fell[i] else ('success' if env.reached[i] else 'timeout')
            episodes.append((o, int(t_ep[i])))
        s, g = env.sample_start_goal(int(done.sum()))
        env.p[done] = s; env.goal[done] = g; env.fell[done] = False; env.reached[done] = False; t_ep[done] = 0
        prev_reset = done.copy()
    else:
        prev_reset = np.zeros(args.E, bool)
    if (t + 1) % 50 == 0:
        print(f'step {t+1} eps={len(episodes)} {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
n = max(1, len(episodes))
cnts = {o: sum(1 for e in episodes if e[0] == o) for o in ('success', 'fall', 'timeout')}
st = [e[1] for e in episodes if e[0] == 'success']
res = dict(args=vars(args), episodes=len(episodes), success=cnts['success'] / n, fall=cnts['fall'] / n, timeout=cnts['timeout'] / n,
           counts=cnts, median_steps=float(np.median(st)) if st else None, episode_list=episodes,
           K_mean=(np.mean(K_hist, 0).tolist() if K_hist else None),
           pred_calls=cnt['calls'], pred_rows=cnt['rows'], pred_rows_per_step=cnt['rows'] / args.total_steps, sec=time.time() - t0)
print(json.dumps({k: v for k, v in res.items() if k not in ('args', 'episode_list')}))
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
