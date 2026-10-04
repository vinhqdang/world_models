"""Train a small JEPA world model on the stochastic navigation data.

python experiments/train_wm.py --kind es --steps 8000 --out runs/es_s0.pt
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.stochnav import collect, render
from selwm.jepa import JEPA

ap = argparse.ArgumentParser()
ap.add_argument('--kind', default='es'); ap.add_argument('--variant', default='cliff')
ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_data', type=int, default=300_000); ap.add_argument('--dim', type=int, default=64)
ap.add_argument('--sigreg', type=float, default=0.09); ap.add_argument('--M', type=int, default=8)
ap.add_argument('--out', default='runs/wm.pt'); ap.add_argument('--device', default='cpu')
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
dev = torch.device(args.device)
torch.manual_seed(args.seed); np.random.seed(args.seed)

cache = f'runs/data_{args.variant}_{args.n_data}.npz'
if os.path.exists(cache):
    d = np.load(cache); P, F, A, P2, F2 = [d[k] for k in ['P', 'F', 'A', 'P2', 'F2']]
else:
    P, F, A, P2, F2 = collect(args.variant, args.n_data, 500, seed=123)
    os.makedirs('runs', exist_ok=True); np.savez(cache, P=P, F=F, A=A, P2=P2, F2=F2)
print('data', P.shape, 'fall transitions %.3f' % (F2 & ~F).mean(), flush=True)
n = len(P); ntr = int(0.95 * n)
T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
P, A, P2 = T(P), T(A), T(P2); F, F2 = T(F, torch.bool), T(F2, torch.bool)

model = JEPA(args.kind, args.dim, sigreg_weight=args.sigreg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(0, ntr, (args.bs,), device=dev)
    x0 = render(P[idx], args.variant, F[idx]); x1 = render(P2[idx], args.variant, F2[idx])
    loss, info = model.loss(x0, A[idx], x1, M=args.M)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f'step {step:6d} loss {float(loss):.4f} pred {info["pred"]:.4f} sigreg {info["sigreg"]:.4f}  {time.time()-t0:.0f}s', flush=True)
torch.save(dict(state=model.state_dict(), args=vars(args)), args.out)
print('saved', args.out)
