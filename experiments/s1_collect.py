"""S1 Phase A collection: real CEM populations on cliff_hi with J8 / J64 model scores and true-simulator costs.

python experiments/s1_collect.py --ckpt suite_hi/ckpt/es_s0.pt --k 0 --scheme indep --n 1000 --out results/s1/pops_es_s0_indep.npz
Per state and population we store: J8 (planner score, M=8), J64 (M=64 model score, fresh noise), pf8 (model failure fraction),
JtA, JtB (two independent 200-particle true costs), pfA, pfB (true pit-fall probability estimates).
Populations: 'f32' final CEM-32 pop (iteration 3), 'i32' iteration-1 pop of the same run, 'f64' final pop of CEM-64, 'i64' its iteration-1 pop.
"""
import argparse, sys, time
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm import s1_pop as S

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--k', type=int, required=True)
ap.add_argument('--scheme', default='indep'); ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--M', type=int, default=8); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--bs', type=int, default=25); ap.add_argument('--out', required=True)
args = ap.parse_args()

mem, anchor, lm = S.load_model(args.ckpt, args.scheme)
P0, G, is_start = S.make_states(args.n, 600 + args.k)          # state seeds 600,601,602 (disjoint from earlier phases)
gen = torch.Generator().manual_seed(650 + args.k)               # planner / model noise
gen_t = torch.Generator().manual_seed(700 + args.k)             # true-simulator noise
res = {}
t0 = time.time()
def put(name, key, arr):
    res.setdefault(f'{name}_{key}', []).append(arr.numpy())
for lo in range(0, args.n, args.bs):
    sl = slice(lo, min(args.n, lo + args.bs))
    p0 = torch.tensor(P0[sl]); g = torch.tensor(G[sl])
    with torch.no_grad():
        z0 = lm.obs_to_latent(p0, torch.zeros(len(p0), dtype=torch.bool)); zg = lm.obs_to_latent(g)
    for N, tag in ((32, '32'), (64, '64')):
        pops = S.cem_pops(lm, z0, zg, N, args.M, args.iters, gen, keep=(0, args.iters - 1) if N == 32 else (args.iters - 1,))
        for it, d in pops.items():
            name = ('f' if it == args.iters - 1 else 'i') + tag
            # J64: re-score the same plans with 64 fresh particles (model expectation, almost noise free)
            if hasattr(lm, 'begin_replan'): lm.begin_replan()
            c64 = torch.cat([lm.cost(lm.rollout(z0[j:j+5], d['acts'][j:j+5], 64, gen, zg[j:j+5]), zg[j:j+5]) for j in range(0, len(z0), 5)])
            jtA, pfA = S.true_cost(lm, mem, anchor, p0, zg, d['acts'], S.MT, gen_t)
            jtB, pfB = S.true_cost(lm, mem, anchor, p0, zg, d['acts'], S.MT, gen_t)
            for key, v in (('J8', d['J']), ('pf8', d['pf']), ('J64', c64.mean(-1)), ('JtA', jtA), ('JtB', jtB), ('pfA', pfA), ('pfB', pfB)):
                put(name, key, v)
    if (lo // args.bs) % 4 == 0:
        print(lo, round(time.time() - t0, 1), flush=True)
out = {k: np.concatenate(v) for k, v in res.items()}
out['P0'], out['G'], out['is_start'] = P0, G, is_start
np.savez_compressed(args.out, **out)
print('done', time.time() - t0)
