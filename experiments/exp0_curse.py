"""Exp0: does the selection gap J_true - J_pred of the chosen plan grow with search budget N?"""
import sys, time, json, torch
sys.path.insert(0, '.')
from selwm.envs import WallMaze
from selwm.models import Ens, fit
from selwm.planners import cem

torch.set_num_threads(4)
env = WallMaze()
res = []
for n_data in [20000, 100000]:
    gen = torch.Generator().manual_seed(0)
    X, A, Xn = env.collect(n_data, gen)
    torch.manual_seed(1)
    model = fit(Ens(K=5), X, A, Xn, epochs=8000)
    with torch.no_grad():
        te = env.collect(5000, torch.Generator().manual_seed(9))
        err = ((model(te[0], te[1]).mean(0) - te[2]) ** 2).sum(-1).mean().item()
    for H in [30]:
        for N in [8, 32, 128, 512, 2048]:
            g2 = torch.Generator().manual_seed(123)
            s, g = env.sample_start_goal(30, g2)
            t0 = time.time()
            a_star, jhat, _ = cem(model, s, g, H=H, N=N, gen=torch.Generator().manual_seed(5))
            xs = env.rollout(s, a_star)
            jtrue = env.cost(xs[:, -1], g)
            r = dict(n_data=n_data, H=H, N=N, onestep_mse=err,
                     jhat=jhat.mean().item(), jtrue=jtrue.mean().item(),
                     gap=(jtrue - jhat).mean().item(),
                     reach=(jtrue.sqrt() < 0.1).float().mean().item(), sec=time.time() - t0)
            res.append(r)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
json.dump(res, open('results/exp0_curse_big.json', 'w'))
