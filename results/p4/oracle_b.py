import numpy as np, torch, pilot_b as pb
def f_true(x, a):
    sn = torch.clamp(x + pb.STEP * a, -1, 1)
    cross = (x[..., 0] * sn[..., 0] < 0) | (sn[..., 0].abs() < 1e-9)
    t = torch.where((sn[..., 0] - x[..., 0]).abs() > 1e-12, -x[..., 0] / (sn[..., 0] - x[..., 0] + 1e-12), torch.zeros_like(x[..., 0]))
    yc = x[..., 1] + t * (sn[..., 1] - x[..., 1])
    blocked = cross & (yc.abs() < 0.4)
    return torch.where(blocked[..., None], x, sn)
class T:
    mean = staticmethod(f_true)
tasks = [pb.sample_task(np.random.RandomState(9999 + i)) for i in range(80)]
print('oracle true-model success', pb.evaluate(T, tasks, 0))
# also: no-wall model (what a perfectly smooth/leaky model would believe)
class NW:
    mean = staticmethod(lambda x, a: torch.clamp(x + pb.STEP * a, -1, 1))
print('no-wall model success', pb.evaluate(NW, tasks, 0))
