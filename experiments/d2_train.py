"""Multi-step trajectory-ES training.  python experiments/d2_train.py --seed 0 --init suite_hi/ckpt/es_s0.pt --out runs/d2/x.pt"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
from selwm.jepa import JEPA
from selwm.d2_windows import collect_windows, pit_distance
from selwm.d2_train import window_loss, encode_windows
from selwm.stochnav import StochNav

ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='cliff_hi'); ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--K', type=int, default=10)
ap.add_argument('--M', type=int, default=8); ap.add_argument('--n_win', type=int, default=60000)
ap.add_argument('--joint_w', type=float, default=1.0); ap.add_argument('--marg_w', type=float, default=1.0); ap.add_argument('--one_w', type=float, default=0.0)
ap.add_argument('--haz', type=float, default=0.0, help='hazard sampling: weight 1+haz for windows that pass within --haz_r of the pit or fall')
ap.add_argument('--haz_r', type=float, default=0.12)
ap.add_argument('--init', default='', help='warm-start checkpoint (one-step ES)'); ap.add_argument('--out', default='runs/d2/wm.pt')
ap.add_argument('--bench', type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
cache = f'runs/d2/windows_{args.variant}_{args.n_win}_K{args.K}.npz'
if os.path.exists(cache):
    d = np.load(cache); P, F, A = d['P'], d['F'], d['A']
else:
    P, F, A = collect_windows(args.variant, args.n_win, args.K); np.savez(cache, P=P, F=F, A=A)
env = StochNav(args.variant, 1, 0)
dist = pit_distance(P.astype(np.float64), F, env.pit).min(1)
hazw = np.where((dist < args.haz_r) | F.any(1), 1.0 + args.haz, 1.0); hazw = hazw / hazw.sum()
print('windows', P.shape, 'frac fall %.3f near-hazard %.3f' % (F[:, -1].mean(), (dist < args.haz_r).mean()), flush=True)
Pt, Ft, At = torch.tensor(P), torch.tensor(F), torch.tensor(A)
Z = encode_windows(Pt, Ft, args.variant, 'fixed')                                 # (W,K+1,3)
ntr = len(P); probs = torch.tensor(hazw)
model = JEPA('es', 3, noise_dim=8, sigreg_weight=0.0, obs='fixed')
if args.init:
    model.load_state_dict(torch.load(args.init)['state'])
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = torch.multinomial(probs, args.bs, replacement=True) if args.haz > 0 else torch.randint(0, ntr, (args.bs,))
    loss, info = window_loss(model, Z[idx], At[idx], args.M, args.joint_w, args.marg_w, args.one_w)
    loss = loss.mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 250 == 0 or step == 1:
        print(f'step {step:6d} loss {float(loss):.4f} {info} {time.time()-t0:.0f}s', flush=True)
    if args.bench and step == args.bench:
        print('sec/step', (time.time() - t0) / step); sys.exit()
cfg = dict(kind='es', variant=args.variant, dim=3, sigreg=0.0, obs='fixed', noise_dim=8, **{k: v for k, v in vars(args).items() if k not in ('variant',)})
torch.save(dict(state=model.state_dict(), args=cfg), args.out)
print('saved', args.out)
