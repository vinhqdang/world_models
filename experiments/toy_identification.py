"""Does dithered, executed-plan-only feedback recover the true optimism curve G(KL)?

Toy: WallMaze + learned MLP ensemble. For every round we compute (oracle) the true cost J_k of the
plan at every CEM iterate k, and compare the oracle curve with the estimate built from ONE randomly
chosen iterate per round.
"""
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from selwm.envs import WallMaze
from selwm.models import Ens, fit
from selwm.pressure import cem_trajectory, OptimismCurve

torch.set_num_threads(4)
env = WallMaze()
gen = torch.Generator().manual_seed(0)
X, A, Xn = env.collect(5000, gen)
torch.manual_seed(1)
model = fit(Ens(K=5), X, A, Xn, epochs=2500)

H, N, ITERS, R = 30, 256, 10, 400
s, g = env.sample_start_goal(R, torch.Generator().manual_seed(11))

def pred_cost(acts):  # (B,N,H,2) -> (B,N)
    B, Nn = acts.shape[:2]
    x = s_cur[:, None].expand(B, Nn, 2).reshape(1, B * Nn, 2).expand(model.K, -1, -1)
    a = acts.reshape(1, B * Nn, H, 2).expand(model.K, -1, -1, -1)
    for t in range(H):
        x = model(x, a[:, :, t])
    gg = g_cur[:, None].expand(B, Nn, 2).reshape(1, B * Nn, 2)
    return ((x - gg) ** 2).sum(-1).mean(0).reshape(B, Nn)

mus_all, kls_all, chat_all, J_all = [], [], [], []
for i in range(0, R, 100):
    s_cur, g_cur = s[i:i + 100], g[i:i + 100]
    with torch.no_grad():
        mus, kls = cem_trajectory(pred_cost, 100, H, 2, N, ITERS, gen=torch.Generator().manual_seed(100 + i))
        # model cost and true cost of every iterate's mean plan
        chat = torch.stack([pred_cost(m[:, None])[:, 0] for m in mus])           # (K+1, B)
        J = torch.stack([env.cost(env.rollout(s_cur, m)[:, -1], g_cur) for m in mus])
    mus_all.append(mus); kls_all.append(kls); chat_all.append(chat); J_all.append(J)
kls = torch.cat(kls_all, 1).numpy(); chat = torch.cat(chat_all, 1).numpy(); J = torch.cat(J_all, 1).numpy()
c0 = chat[0]                                                                       # cost of prior mean plan
e_rel = (J - chat) / np.maximum(c0, 1e-6)                                          # normalised optimism
print('iterate  mean KL   c_hat    J_true   rel-optimism   (oracle, all rounds)')
for k in range(ITERS + 1):
    print(f'{k:7d} {kls[k].mean():8.1f} {chat[k].mean():8.4f} {J[k].mean():8.4f} {e_rel[k].mean():10.3f}')
best_k_oracle = J.mean(1).argmin()
print('oracle best fixed iterate (min mean true cost):', best_k_oracle, ' vs last iterate', ITERS)

# online estimator: ONE random iterate per round, only its outcome revealed
rng = np.random.default_rng(0)
grid = np.arange(1, ITERS + 1)
res = []
for n_rounds in [25, 50, 100, 200, 400]:
    cur = OptimismCurve()
    for r in range(n_rounds):
        k = rng.choice(grid)
        cur.add(kls[k, r], e_rel[k, r])
    cur.fit()
    est = cur(kls[grid].reshape(-1))
    tru = e_rel[grid].reshape(-1)
    # compare estimated curve to oracle curve through kl-binned oracle means
    kb = np.quantile(kls[grid].reshape(-1), np.linspace(0, 1, 11))
    idx = np.digitize(kls[grid].reshape(-1), kb[1:-1])
    orc = np.array([tru[idx == b].mean() for b in range(10)]); es = np.array([est[idx == b].mean() for b in range(10)])
    # decision: pick iterate minimising estimated true cost  c_hat_k + c0 * G(kl_k), compare realised J of that rule
    pick = np.argmin(chat[grid] + c0[None] * cur(kls[grid]), axis=0) + 1             # (R,)
    J_rule = J[pick, np.arange(R)].mean()
    res.append((n_rounds, float(np.sqrt(((orc - es) ** 2).mean())), float(J_rule)))
    print(f'rounds={n_rounds:4d}  RMSE(est curve vs oracle curve)={res[-1][1]:.4f}   mean true cost of rule={J_rule:.4f}')
print('baselines: always last iterate J=%.4f ; best fixed iterate J=%.4f ; per-round oracle-best iterate J=%.4f' %
      (J[ITERS].mean(), J.mean(1).min(), J.min(0).mean()))
json.dump(dict(kls=kls.mean(1).tolist(), chat=chat.mean(1).tolist(), J=J.mean(1).tolist(), res=res), open('results/toy_identification.json', 'w'))
