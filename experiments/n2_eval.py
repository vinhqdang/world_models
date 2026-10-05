"""N2 closed-loop evaluation (open-loop CEM, cliff_hi, es models).  Same protocol as experiments/d8_eval.py (E=16, N=32, H=10,
3 CEM iterations, 120-step streaming episodes, paired start/goal per (seed, slot, episode index)) with a --scheme argument:
any name of selwm.d6_model.SCHEMES (iid / CRN / chained CRN) or an n2 scheme (selwm.n2_model.make_n2_model)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe
from selwm.risk_plan import cem
from selwm.noise_aware import NoiseAwareScorer
from selwm.d6_model import CoupledLatentModel, make_noise
from selwm.n2_model import make_n2_model

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--scheme', required=True)
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--E', type=int, default=16)
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--H', type=int, default=10); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--total_steps', type=int, default=400); ap.add_argument('--ep_len', type=int, default=120)
ap.add_argument('--kappa', type=float, default=3.0); ap.add_argument('--stage_w', type=float, default=1.0)
ap.add_argument('--learned_path', default=None)
ap.add_argument('--seed', type=int, default=500); ap.add_argument('--out', required=True)
args = ap.parse_args()
dev = torch.device('cpu')
np.random.seed(args.seed); torch.manual_seed(args.seed)
ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a.get('noise_dim', 8), sigreg_weight=a['sigreg'], obs=a.get('obs', 'pixel')).to(dev)
mem.load_state_dict(ck['state']); mem.eval()

def start_goal_scale():
    e = StochNav(args.variant, 1, 4242)
    sp, gp = e.sample_start_goal(512)
    with torch.no_grad():
        zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, args.variant, mem.obs))
        zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, args.variant, mem.obs))
    return float(((zs - zg_) ** 2).sum(-1).median())

anchor = FailureAnchor(mem.encode, args.variant, mem.obs, device=dev)
kw = dict(stochastic=True, stage_w=args.stage_w, anchor=anchor, kappa=args.kappa, scale=start_goal_scale())
lm = make_n2_model(args.scheme, mem, args.variant, learned_path=args.learned_path, **kw)
if lm is None:
    lm = CoupledLatentModel(mem, args.variant, noise=make_noise(args.scheme), **kw)

E = args.E
env = StochNav(args.variant, E, 0)
env.rng = np.random.default_rng([7000, args.seed])
slot_env = [StochNav(args.variant, 1, 0) for _ in range(E)]
for i, se in enumerate(slot_env): se.rng = np.random.default_rng([8000, args.seed, i])
def draw(i):
    s, g = slot_env[i].sample_start_goal(1); return s[0], g[0]
S0, G0 = zip(*[draw(i) for i in range(E)])
env.reset(np.array(S0), np.array(G0))
ep_idx = np.zeros(E, int)
gen = torch.Generator(device=dev).manual_seed(args.seed)
t_ep = np.zeros(E, int)
prev_reset = np.zeros(E, bool)
eps = []
t0 = time.time()
for t in range(args.total_steps):
    p = torch.tensor(env.p, dtype=torch.float32); fell = torch.tensor(env.fell)
    z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32))
    if hasattr(lm, 'begin_replan'): lm.begin_replan(reset_mask=prev_reset)
    elif hasattr(lm, 'noise'): lm.noise.begin_replan(prev_reset)
    plan = cem(lm, z0, zg, args.H, args.N, args.M, iters=args.iters, gen=gen, score_fn=NoiseAwareScorer(0.0, 0.1, False))
    env.step(plan[:, 0].cpu().numpy()); t_ep += 1
    done = env.fell | env.reached | (t_ep >= args.ep_len)
    if done.any():
        for i in np.where(done)[0]:
            o = 'fall' if env.fell[i] else ('success' if env.reached[i] else 'timeout')
            eps.append((int(i), int(ep_idx[i]), o, int(t_ep[i])))
            ep_idx[i] += 1
            s, g = draw(i)
            env.p[i] = s; env.goal[i] = g; env.fell[i] = False; env.reached[i] = False; t_ep[i] = 0
        prev_reset = done.copy()
    else:
        prev_reset = np.zeros(E, bool)
n = len(eps)
cnt = {o: sum(1 for e in eps if e[2] == o) for o in ('success', 'fall', 'timeout')}
res = dict(args=vars(args), episodes=n, success=cnt['success'] / max(1, n), fall=cnt['fall'] / max(1, n), timeout=cnt['timeout'] / max(1, n),
           sec=time.time() - t0, records=eps)
print(json.dumps({k: v for k, v in res.items() if k not in ('args', 'records')}))
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True); json.dump(res, open(args.out, 'w'))
