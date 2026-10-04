import sys, json, torch
sys.path.insert(0, '.')
from selwm.envs import WallMaze
from selwm.planners import cem
torch.set_num_threads(4)
env = WallMaze()

class Oracle:
    K = 1
    def __call__(self, x, a):
        return env.transition(x, a)

for H in [30, 40]:
    for N in [128, 512, 2048]:
        s, g = env.sample_start_goal(30, torch.Generator().manual_seed(123))
        a, jhat, _ = cem(Oracle(), s, g, H=H, N=N, gen=torch.Generator().manual_seed(5))
        jt = env.cost(env.rollout(s, a)[:, -1], g)
        print(json.dumps(dict(H=H, N=N, jhat=round(jhat.mean().item(), 4), jtrue=round(jt.mean().item(), 4),
                              reach=round((jt.sqrt() < 0.1).float().mean().item(), 3))), flush=True)
