"""EXPLORATORY pilot: cache cross-fitted residuals + eval tensors for a cell."""
import sys, glob, pickle, time
sys.path.insert(0, "/home/user/world_models"); sys.path.insert(0, "/home/user/world_models/experiments")
import numpy as np, torch
from cwk import upstream, cells, worlds, runner as RN, metrics as M, predictor as P, kernels as KR, evalpop as EP
DATA = "/home/user/world_models/results/cwk/data"
name, seed = sys.argv[1], int(sys.argv[2])
w = cells.STAGE2[name]()
col = worlds.load(glob.glob(f"{DATA}/{name}_col_*.npz")[0]); pop = worlds.load(glob.glob(f"{DATA}/{name}_pop_512x64.npz")[0])
cell = cells.pipeline_cell(w, col, pop, None)
torch.set_num_threads(2)
t0 = time.time()
truth = np.asarray(pop["truth"]); N, E, K, D = truth.shape; coords = cell.coords; d = len(coords)
T = col["actions"].shape[1]
H, a, y, s, ep, st = P.flatten_transitions(col)
cf = P.crossfit(H, a, y, s, ep, seed, epochs=20, width=384, batch=512)
full = cf["full"]; mu_pop = EP.mean_predictions(full, pop); sc = full.sc
kd = KR.prepare(H, a, y, s, ep, st, cf["oof"], coords, cell.axes, cell.a_lim, sc, T)
Hn, s_pop, A = EP.pop_inputs(pop)
hs = torch.as_tensor((np.repeat(Hn, E, 0) - sc.hm) / sc.hs, dtype=torch.float32)
psi_pop = [torch.as_tensor(np.repeat(KR.psi_features(A[:, k], cell.axes, cell.a_lim)[0], E, 0), dtype=torch.float32) for k in range(K)]
r_std = torch.as_tensor(((truth[:, :, 0][..., coords] - mu_pop[:, 0][:, None, coords]) / kd.sigma).reshape(N * E, d), dtype=torch.float32)
det = RN.deterministic(cell)
pickle.dump(dict(kd=kd, hs=hs, psi_pop=psi_pop, r_std=r_std, mu_pop=mu_pop, truth=truth, weights=pop["weights"], coords=coords,
                 groups=cell.groups, primary=cell.primary, det=det, axes=cell.axes, a_lim=cell.a_lim), open(f"{name}_s{seed}.pkl", "wb"))
print(name, "prep", time.time() - t0)
