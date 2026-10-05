"""N2: learn a shared (coupled) node design q[m, t] (M=8 paths x H=10 steps, coordinates along the model's active noise direction
vbar; the 7 complementary coordinates are a fixed random draw) that minimises the ranking error of the M-path estimator against a
high-budget reference, over a distribution of (scenario, candidate-set) pairs.

Training data: model 0 only, scenario seeds 9000+ (disjoint from the ranking-benchmark seeds 5000+ and from all closed-loop seeds).
Loss: MSE between centred estimator costs and centred reference costs (hard-failure reference; the estimator uses a smooth failure
indicator sigmoid((tau - d)/s) and a noisy-or running maximum so that gradients flow).  The learned q is then used with any model
through its own vbar (selwm.n2_model.N2Noise kind 'learned') and evaluated with the hard cost.
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import StochNav
from selwm.jepa import JEPA, FailureAnchor, observe, LatentModel
from selwm.d6_model import CoupledLatentModel
from selwm.d6_noise import NoiseSource
from selwm.n2_model import active_direction, oa_design

ap = argparse.ArgumentParser()
ap.add_argument('--S', type=int, default=48); ap.add_argument('--N', type=int, default=32); ap.add_argument('--Mref', type=int, default=512)
ap.add_argument('--iters', type=int, default=250); ap.add_argument('--bs', type=int, default=6); ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--model', type=int, default=0); ap.add_argument('--seed', type=int, default=9000)
ap.add_argument('--out', default='results/n2/learned_nodes.pt'); ap.add_argument('--M', type=int, default=8); ap.add_argument('--H', type=int, default=10)
args = ap.parse_args()
torch.manual_seed(args.seed)
variant = 'cliff_hi'
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location('n2rank_helpers', 'experiments/n2_rank_helpers.py')
H_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(H_)

ck = torch.load(f'suite_hi/ckpt/es_s{args.model}.pt', map_location='cpu'); a = ck['args']
mem = JEPA(a['kind'], a['dim'], noise_dim=a['noise_dim'], sigreg_weight=a['sigreg'], obs=a['obs']); mem.load_state_dict(ck['state']); mem.eval()
for p_ in mem.parameters(): p_.requires_grad_(False)
anchor = FailureAnchor(mem.encode, variant, mem.obs)
e = StochNav(variant, 1, 4242); sp, gp = e.sample_start_goal(512)
with torch.no_grad():
    zs = mem.encode(observe(torch.tensor(sp, dtype=torch.float32), None, variant, mem.obs)); zg_ = mem.encode(observe(torch.tensor(gp, dtype=torch.float32), None, variant, mem.obs))
scale = float(((zs - zg_) ** 2).sum(-1).median())
kw = dict(stage_w=1.0, anchor=anchor, kappa=3.0, scale=scale)
vbar = active_direction(mem)
ref_model = CoupledLatentModel(mem, variant, noise=NoiseSource(gen_kind='sobol80', anti=True, share=False), **kw)

# ---- training scenarios (both regimes), reference costs
rng = np.random.default_rng(args.seed); rng_t = torch.Generator().manual_seed(args.seed)
sets = []
t0 = time.time()
for regime in ('wide', 'local'):
    p, g = H_.scenarios(args.S, rng, variant)
    acts = H_.candidates(p, g, args.N, args.H, regime, rng_t)
    z0 = ref_model.obs_to_latent(p); zg = ref_model.obs_to_latent(g)
    gen = torch.Generator().manual_seed(4321)
    cs = []
    for i in range(0, args.S, 2):
        cs.append(ref_model.cost(ref_model.rollout(z0[i:i + 2], acts[i:i + 2], args.Mref, gen, zg[i:i + 2]), zg[i:i + 2]).mean(-1))
    sets.append((z0, zg, acts, torch.cat(cs)))
    print('reference', regime, f'{time.time() - t0:.0f}s', flush=True)

# ---- smooth estimator
tau = anchor.tau; sm = 0.05 * tau
def soft_cost(z0, zg, acts, U):
    """z0 (B,D) zg (B,D) acts (B,N,H,A) U (M,H,K) -> (B,N) mean over M paths."""
    B, N, Hh, A = acts.shape; M = U.shape[0]
    z = z0[:, None, None, :].expand(B, N, M, z0.shape[-1])
    acc = torch.zeros(B, N, M); fl = torch.zeros(B, N, M)
    for t in range(Hh):
        a = acts[:, :, t, None, :].expand(B, N, M, A)
        u = U[None, None, :, t, :].expand(B, N, M, U.shape[-1])
        z = mem.pred(z, a, u)[0]
        acc = acc + ((z - zg[:, None, None, :]) ** 2).sum(-1) / Hh
        hit = torch.sigmoid((tau - ((z - anchor.z) ** 2).sum(-1)) / sm)
        fl = fl + (1 - fl) * hit
    c = (((z - zg[:, None, None, :]) ** 2).sum(-1) + 1.0 * acc) / scale + 3.0 * fl
    return c.mean(-1)

gx = torch.Generator().manual_seed(2024)
xi = torch.randn(args.M, args.H, 8, generator=gx)
xi = xi - (xi * vbar).sum(-1, keepdim=True) * vbar
q = torch.tensor(1.5 * oa_design(args.M, args.H, 0), dtype=torch.float32).clone().requires_grad_(True)       # init: sign design
opt = torch.optim.Adam([q], lr=args.lr)
def U_of(q):
    return q[..., None] * vbar + xi

def loss_batch(idx_set, idx_sc):
    z0, zg, acts, refc = sets[idx_set]
    c = soft_cost(z0[idx_sc], zg[idx_sc], acts[idx_sc], U_of(q))
    r = refc[idx_sc]
    return (((c - c.mean(-1, keepdim=True)) - (r - r.mean(-1, keepdim=True))) ** 2).mean()

with torch.no_grad():
    base = np.mean([float(loss_batch(s, torch.arange(args.S))) for s in range(2)])
print('initial (sign design) loss', base, flush=True)
# hold out the last 12 scenarios of each regime for model selection
tr = torch.arange(0, args.S - 12); va = torch.arange(args.S - 12, args.S)
best = (1e9, q.detach().clone())
for it in range(args.iters):
    s = it % 2
    idx = tr[torch.randperm(len(tr))[:args.bs]]
    loss = loss_batch(s, idx)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 25 == 24 or it == args.iters - 1:
        with torch.no_grad():
            v = np.mean([float(loss_batch(s_, va)) for s_ in range(2)])
        if v < best[0]: best = (v, q.detach().clone())
        print(f'it {it + 1} train {float(loss):.4f} val {v:.4f} best {best[0]:.4f} {time.time() - t0:.0f}s', flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(dict(q=best[1], xi=xi, M=args.M, H=args.H, val=best[0], init_val=base), args.out.replace('.pt', '_q.pt'))
print('saved q; |q| mean', best[1].abs().mean().item())
