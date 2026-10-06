"""P3 analysis: rank correlation of offline statistics with ground-truth closed-loop outcomes across candidate models.

Conventions fixed before looking at the data: every statistic is turned into a QUALITY score (higher = predicted better):
  error-type statistics (ES, MSE, NLL, Brier, RMS, pick cost, regret, optimism, OW-*): quality = -stat
  rank agreement (tau_cell): quality = +stat;  imagined closed loop (direct method): quality = +im_succ for success, -im_fall for fall.
Targets: success rate (higher better) and fall rate (lower better, so correlated with -fall).  tau > 0 always means 'predicts correctly'.
Bootstrap CIs resample MODELS (2000 draws).
"""
import json, glob, os, sys, itertools
import numpy as np
from scipy.stats import kendalltau, spearmanr

OUT = 'results/p3'
gt, st = {}, {}
for f in glob.glob(f'{OUT}/gt/*.json'):
    r = json.load(open(f)); gt[os.path.basename(f)[:-5]] = r
for f in glob.glob(f'{OUT}/stats/*.json'):
    st[os.path.basename(f)[:-5]] = json.load(open(f))
names = sorted(set(gt) & set(st))
kind = np.array([st[n]['kind'] for n in names])
succ = np.array([gt[n]['success'] for n in names]); fall = np.array([gt[n]['fall'] for n in names])
nep = np.array([gt[n]['episodes'] for n in names])
print(f'{len(names)} models with both ground truth and statistics; kinds', {k: int((kind == k).sum()) for k in set(kind)})

LOWER = ['ES1', 'MSE1', 'NLL1', 'ES_H', 'SE_H', 'ES1_edge', 'Brier_hazard', 'Brier_hazard_edge', 'pool_rmse',
         'pool_pick_gain', 'pool_pick_y', 'pool_pick_regret', 'pool_pick_opt', 'pool_pick_fall_gain', 'OW_ES1', 'OW_Brier_hazard']
HIGHER = ['pool_tau_cell']
GROUP = {'ES1': 'baseline', 'MSE1': 'baseline', 'NLL1': 'baseline', 'ES_H': 'baseline', 'SE_H': 'baseline', 'ES1_edge': 'baseline (region)',
         'Brier_hazard': 'baseline', 'Brier_hazard_edge': 'baseline (region)', 'pool_rmse': 'plan-cost error (planner functional)',
         'pool_tau_cell': 'idea1 plan-pool replay', 'pool_pick_gain': 'idea1 plan-pool replay', 'pool_pick_y': 'idea1 plan-pool replay',
         'pool_pick_regret': 'idea1 plan-pool replay', 'pool_pick_opt': 'idea3 selected-plan optimism', 'pool_pick_fall_gain': 'idea1 plan-pool replay',
         'OW_ES1': 'idea2 occupancy-weighted', 'OW_Brier_hazard': 'idea2 occupancy-weighted', 'im_succ': 'direct method (imagined closed loop)', 'im_fall': 'direct method (imagined closed loop)'}


def quality(stat, target):
    if stat == 'im_succ': return np.array([st[n]['im_succ'] for n in names]) if target == 'succ' else None
    if stat == 'im_fall': return -np.array([st[n]['im_fall'] for n in names]) if target == 'fall' else None
    x = np.array([st[n][stat] for n in names], float)
    return -x if stat in LOWER else x


def tau(x, y, mask=None):
    if mask is not None: x, y = x[mask], y[mask]
    if len(x) < 4 or np.std(x) == 0: return np.nan
    return kendalltau(x, y)[0]


def within_kind_tau(x, y):
    c = d = 0
    for k in set(kind):
        idx = np.where(kind == k)[0]
        for i, j in itertools.combinations(idx, 2):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0; d += s < 0
    return (c - d) / max(c + d, 1)


def boot(fn, x, y, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = fn(x[i], y[i])
        if np.isfinite(v): vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def split_half_tau():
    """Reliability of the ground truth itself: success rate from even vs odd env slots."""
    a, b, fa, fb = [], [], [], []
    for n in names:
        rec = gt[n]['records']
        e = [r for r in rec if r[0] % 2 == 0]; o = [r for r in rec if r[0] % 2 == 1]
        a.append(np.mean([r[2] == 'success' for r in e])); b.append(np.mean([r[2] == 'success' for r in o]))
        fa.append(-np.mean([r[2] == 'fall' for r in e])); fb.append(-np.mean([r[2] == 'fall' for r in o]))
    return kendalltau(a, b)[0], kendalltau(fa, fb)[0]


stats = [s for s in LOWER + HIGHER + ['im_succ', 'im_fall'] if all(s in st[n] for n in names)]
rows = []
for s in stats:
    for tgt, y in (('succ', succ), ('fall', -fall)):
        q = quality(s, tgt)
        if q is None: continue
        r = dict(stat=s, group=GROUP.get(s, ''), target=tgt)
        r['tau_all'] = tau(q, y); lo, hi = boot(lambda a, b: kendalltau(a, b)[0], q, y); r['tau_all_ci'] = (lo, hi)
        r['rho_all'] = spearmanr(q, y)[0]
        r['tau_within_kind'] = within_kind_tau(q, y)
        r['tau_es'] = tau(q, y, kind == 'es'); r['tau_gauss'] = tau(q, y, kind == 'gauss'); r['tau_det'] = tau(q, y, kind == 'det')
        r['tau_nodet'] = tau(q, y, kind != 'det')
        rows.append(r)

sh = split_half_tau()
lines = []
lines.append(f'# P3 results table (n_models={len(names)}; ground truth: fixed open-loop CEM planner, eval seed 901, {nep.min()}-{nep.max()} episodes/model)\n')
lines.append(f'Ground-truth reliability (Kendall tau between even-slot and odd-slot halves): success {sh[0]:.2f}, fall {sh[1]:.2f}. This bounds what any statistic can reach.\n')
lines.append('GT summary by kind: ' + '; '.join(f'{k}: succ {succ[kind==k].mean():.3f} (range {succ[kind==k].min():.2f}-{succ[kind==k].max():.2f}), fall {fall[kind==k].mean():.3f} (range {fall[kind==k].min():.2f}-{fall[kind==k].max():.2f}), n={int((kind==k).sum())}' for k in sorted(set(kind))) + '\n')
for tgt, title in (('succ', 'Target: closed-loop SUCCESS rate'), ('fall', 'Target: closed-loop FALL rate (tau with -fall)')):
    lines.append(f'\n## {title}\n')
    lines.append('| statistic | group | tau (all) [95% CI over models] | Spearman | tau within kind | tau es-only | tau gauss-only | tau det-only | tau non-det |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in sorted([r for r in rows if r['target'] == tgt], key=lambda r: -np.nan_to_num(r['tau_all'], nan=-9)):
        f = lambda v: 'n/a' if not np.isfinite(v) else f'{v:+.2f}'
        lines.append(f"| {r['stat']} | {r['group']} | {f(r['tau_all'])} [{f(r['tau_all_ci'][0])}, {f(r['tau_all_ci'][1])}] | {f(r['rho_all'])} | {f(r['tau_within_kind'])} | {f(r['tau_es'])} | {f(r['tau_gauss'])} | {f(r['tau_det'])} | {f(r['tau_nodet'])} |")
# per-model table
lines.append('\n## Per-model data\n')
cols = ['ES1', 'MSE1', 'NLL1', 'ES_H', 'pool_pick_gain', 'pool_pick_opt', 'OW_ES1', 'im_succ', 'im_fall']
lines.append('| model | kind | steps | n_data | succ | fall | ' + ' | '.join(cols) + ' |')
lines.append('|---|---|---|---|---|---|' + '---|' * len(cols))
for i, n in enumerate(names):
    lines.append(f"| {n} | {kind[i]} | {st[n]['steps']} | {st[n]['n_data']} | {succ[i]:.3f} | {fall[i]:.3f} | " + ' | '.join(f"{st[n].get(c, float('nan')):.3f}" for c in cols) + ' |')
open(f'{OUT}/RESULTS_TABLE.md', 'w').write('\n'.join(lines))
print('\n'.join(lines))
json.dump(dict(names=names, rows=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]), open(f'{OUT}/correlations.json', 'w'), default=float)
