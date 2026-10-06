"""P3: top-1 model selection regret and kind-only control, from the same data as p3_analyze.py."""
import json, glob, os, numpy as np
from scipy.stats import kendalltau
gt = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob('results/p3/gt/*.json')}
st = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob('results/p3/stats/*.json')}
names = sorted(set(gt) & set(st)); kind = np.array([st[n]['kind'] for n in names])
succ = np.array([gt[n]['success'] for n in names]); fall = np.array([gt[n]['fall'] for n in names])
LOWER = ['ES1','MSE1','NLL1','ES_H','SE_H','ES1_edge','Brier_hazard','pool_pick_gain','pool_pick_opt','pool_pick_fall_gain','OW_ES1','OW_Brier_hazard','pool_rmse']
L = ['| statistic | picked model | success of pick | fall of pick | success regret vs best | fall excess vs safest |','|---|---|---|---|---|---|']
def row(nm, q):
    i = int(np.argmax(q)); L.append(f'| {nm} | {names[i]} | {succ[i]:.3f} | {fall[i]:.3f} | {succ.max()-succ[i]:.3f} | {fall[i]-fall.min():.3f} |')
for s in LOWER: row(s + ' (min)', -np.array([st[n][s] for n in names]))
row('pool_tau_cell (max)', np.array([st[n]['pool_tau_cell'] for n in names]))
row('im_succ (max)', np.array([st[n]['im_succ'] for n in names]))
rng = np.random.default_rng(0); L.append(f'| random model (expected) | - | {succ.mean():.3f} | {fall.mean():.3f} | {succ.max()-succ.mean():.3f} | {fall.mean()-fall.min():.3f} |')
L.append(f'| oracle best-success model | {names[int(np.argmax(succ))]} | {succ.max():.3f} | {fall[int(np.argmax(succ))]:.3f} | 0 | - |')
L.append(f'| oracle safest model | {names[int(np.argmin(fall))]} | {succ[int(np.argmin(fall))]:.3f} | {fall.min():.3f} | - | 0 |')
# kind-only control: score = mean GT of the model's kind (an oracle that knows only the architecture family)
km = {k: succ[kind == k].mean() for k in set(kind)}; kf = {k: fall[kind == k].mean() for k in set(kind)}
qs = np.array([km[k] for k in kind]); qf = -np.array([kf[k] for k in kind])
L.append(f'\nKind-only oracle (score = mean ground truth of the model family): tau with success {kendalltau(qs, succ)[0]:+.2f}, tau with -fall {kendalltau(qf, -fall)[0]:+.2f}.')
L.append('Family means: ' + ', '.join(f'{k}: succ {km[k]:.3f} fall {kf[k]:.3f}' for k in sorted(km)))
# quality-of-training control: ordinal "steps x data" within kind is uninformative for success; report spearman of log steps vs GT within kind
for k in sorted(set(kind)):
    idx = kind == k; stp = np.array([st[n]['steps'] for n in names])[idx]
    L.append(f'{k}: Spearman(steps, success) {np.corrcoef(np.argsort(np.argsort(stp)), np.argsort(np.argsort(succ[idx])))[0,1]:+.2f}, Spearman(steps, fall) {np.corrcoef(np.argsort(np.argsort(stp)), np.argsort(np.argsort(fall[idx])))[0,1]:+.2f} (n={idx.sum()})')
open('results/p3/SELECTION.md','w').write('\n'.join(L)); print('\n'.join(L))
