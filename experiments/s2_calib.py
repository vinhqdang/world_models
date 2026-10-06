"""S2 Phase B calibration collection: base planner (open-loop CEM, chained CRN, D9 'ol_crn3') in closed loop on cliff_hi (wind 0.09);
at every visited state we record, for the plan the planner executes (mean of the final elites) and for other plans of the same CEM call,
the planner's model score and the true-simulator open-loop cost / pit-fall probability (200 fresh true particles).

python experiments/s2_calib.py --ckpt suite_hi/ckpt/es_s0.pt --m 0 --N 32 --seed 1000 --steps 150 --out results/s2/calib_N32_m0.npz

Stored per state: j (planner score of executed plan), pf (model fall prob of executed plan, 8 CRN particles), L, pT (true cost, true fall prob),
4 random final-population members (jm, pfm, Lm, pTm), K=4 escalation candidates [executed plan, top-3 members] re-scored with
64 fresh independent particles (jc, pfc) and their true values (Lc, pTc; Lc[:,0]==L), delayed realised outcome F (fall within next H steps of the
executed closed-loop trajectory; -1 if unresolved), state descriptors, and episode ids (slot, j) for episode-level splits.
"""
import argparse, sys, time, json
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_num_threads(1)
from selwm import s2_scc as S
from selwm.s1_pop import true_cost
from selwm.noise_aware import NoiseAwareScorer

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--m', type=int, required=True)
ap.add_argument('--N', type=int, default=32); ap.add_argument('--M', type=int, default=8); ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--seed', type=int, required=True); ap.add_argument('--steps', type=int, default=150); ap.add_argument('--E', type=int, default=16)
ap.add_argument('--Mt', type=int, default=200); ap.add_argument('--Mp', type=int, default=64); ap.add_argument('--out', required=True)
args = ap.parse_args()
assert args.seed >= 1000
np.random.seed(args.seed); torch.manual_seed(args.seed)
mem, anchor, lm, lm_ind = S.load_models(args.ckpt)
cnt = S.count_rows(mem)
E = args.E
env = S.SlotEnv(args.seed, E, wind=0.09, phase=0)
gen = torch.Generator().manual_seed(args.seed); gen_t = torch.Generator().manual_seed(args.seed + 500_000)
sc = NoiseAwareScorer(0.0, 0.1, False)
tracker = S.OutcomeTracker(E)
R = {k: [] for k in ['j', 'pf', 'L', 'pT', 'jm', 'pfm', 'Lm', 'pTm', 'jc', 'pfc', 'Lc', 'pTc', 'pos', 'goal', 't_ep', 'slot', 'jep', 'step', 'js', 'pfs']}
F = []
prev_reset = np.zeros(E, bool); t0 = time.time()
idx_all = np.arange(E)
for t in range(args.steps):
    p = torch.tensor(env.p, dtype=torch.float32); fell = torch.tensor(env.fell)
    z0 = lm.obs_to_latent(p, fell); zg = lm.obs_to_latent(torch.tensor(env.goal, dtype=torch.float32))
    lm.begin_replan(reset_mask=prev_reset)
    pop = S.cem_pop(lm, z0, zg, args.N, args.M, args.iters, gen, sc)
    plan = pop['plan']
    j, pf = S.score_exec(lm, z0, zg, plan, args.M, gen)
    ridx = torch.randint(0, args.N, (E, 4), generator=gen)
    acts_m = torch.gather(pop['acts'], 1, ridx[:, :, None, None].expand(-1, -1, S.H, S.A))
    jm = torch.gather(pop['s'], 1, ridx); pfm = torch.gather(pop['pf'], 1, ridx)
    cand = S.candidates(pop)
    jc, pfc = S.rescore(lm_ind, z0, zg, cand, args.Mp, gen)
    plans = torch.cat([plan[:, None], acts_m, cand[:, 1:]], 1)                     # (E,8): exec, 4 members, top-3 members
    Lt, pT = true_cost(lm, mem, anchor, p, zg, plans, args.Mt, gen_t)
    Lc = torch.cat([Lt[:, :1], Lt[:, 5:]], 1); pTc = torch.cat([pT[:, :1], pT[:, 5:]], 1)
    n0 = len(R['j'])
    alive = ~(env.fell | env.reached)
    for i in range(E):
        k = len(R['j'])
        for key, v in (('j', j[i]), ('pf', pf[i]), ('L', Lt[i, 0]), ('pT', pT[i, 0]), ('jm', jm[i]), ('pfm', pfm[i]), ('Lm', Lt[i, 1:5]), ('pTm', pT[i, 1:5]),
                       ('jc', jc[i]), ('pfc', pfc[i]), ('Lc', Lc[i]), ('pTc', pTc[i]), ('js', pop['s'][i].min()), ('pfs', pop['pf'][i].gather(0, pop['s'][i].argmin()[None])[0])):
            R[key].append(np.asarray(v))
        R['pos'].append(env.p[i].copy()); R['goal'].append(env.goal[i].copy()); R['t_ep'].append(env.t_ep[i]); R['slot'].append(i); R['jep'].append(env.j[i]); R['step'].append(t)
        F.append(-1); tracker.add(i, k)
    env.step(plan[:, 0].numpy(), idx_all)
    fin = env.finished(idx_all); prev_reset = np.zeros(E, bool)
    for i in range(E):
        for k, f in tracker.after_step(i, bool(env.fell[i]), bool(fin[i])): F[k] = f
        if fin[i]:
            env.next_episode(i); prev_reset[i] = True
    if (t + 1) % 25 == 0:
        print(f't={t+1} states={len(F)} {time.time()-t0:.0f}s', flush=True)
out = {k: np.array(v) for k, v in R.items()}
out['F'] = np.array(F)
out['rows_per_step'] = cnt['rows'] / args.steps
np.savez_compressed(args.out, **out)
print('done', len(F), 'states', time.time() - t0, 's', 'rows/step', cnt['rows'] / args.steps)
