"""Generate the LaTeX tables of the manuscript directly from the result files (no numbers are typed by hand).

python paper/make_tables.py   (run from the repository root)
Writes paper/tables/*.tex and paper/tables/numbers.json.
"""
import glob
import json
import os
import re

import numpy as np

rng = np.random.default_rng(0)
os.makedirs('paper/tables', exist_ok=True)
NUM = {}


# ----------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def from_counts(files):
    """Pool several eval_stream result files -> (n, k_success, k_fall, k_timeout, median steps)."""
    n = ks = kf = kt = 0
    steps = []
    for f in files:
        d = json.load(open(f))
        e = d['episodes']
        n += e
        ks += round(d['success'] * e)
        kf += round(d['fall'] * e)
        kt += round(d['timeout'] * e)
        if d.get('median_steps'):
            steps.append(d['median_steps'])
    return n, ks, kf, kt, (float(np.mean(steps)) if steps else float('nan'))


def from_episode_lists(files):
    eps = [tuple(e) for f in files for e in json.load(open(f))['episode_list']]
    n = len(eps)
    o = [e[0] for e in eps]
    st = [e[1] for e in eps if e[0] == 'success']
    return n, o.count('success'), o.count('fall'), o.count('timeout'), (float(np.median(st)) if st else float('nan'))


def cell(k, n):
    lo, hi = wilson(k, n)
    return f'{k / n:.3f} \\,[{lo:.3f},{hi:.3f}]'


def row(name, stats, extra=''):
    n, ks, kf, kt, ms = stats
    return f'{name} & {n} & {cell(ks, n)} & {cell(kf, n)} & {cell(kt, n)} & {ms:.0f}{extra} \\\\'


HEAD = (r'\begin{tabular}{lrcccr}' + '\n' + r'\toprule' + '\n' +
        r'Planner & Episodes & Success & Fall & Timeout & Steps \\' + '\n' + r'\midrule')


def table(rows, path, midrules=()):
    out = [HEAD]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append(r'\midrule')
        out.append(r)
    out += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(out) + '\n')


def key(name, stats):
    n, ks, kf, kt, ms = stats
    NUM[name] = dict(n=n, success=ks / n, fall=kf / n, timeout=kt / n, median_steps=ms)


# ----------------------------------------------------------------------------- main table (identical protocol, eval seed 211)
rows = []
spec = [
    ('Deterministic predictor, mean rollout', 'results/main/det_mean_m*.json', from_counts, 'det'),
    ('Ensemble of 3 deterministic predictors', 'results/main/ens3_m*.json', from_counts, 'ens3'),
    ('Gaussian predictor, particles', 'results/main/gauss_exp_m*.json', from_counts, 'gauss'),
    ('Energy-score predictor, particles (open loop)', 'results/d9/ol_indep_m[0-9].json', from_episode_lists, 'es_ol'),
    ('\\quad + coupled noise (CRN, chained)', 'results/d9/ol_crn3_m[0-9].json', from_episode_lists, 'es_ol_crn'),
    ('\\quad + feedback gains', 'results/d9/fbstep_indep_m[0-9].json', from_episode_lists, 'es_fb'),
    ('\\quad + feedback gains + coupled noise', 'results/d9/fbstep_crn3_m[0-9].json', from_episode_lists, 'es_fb_crn'),
]
for name, pat, fn, k in spec:
    files = sorted(glob.glob(pat))
    if not files:
        continue
    st = fn(files)
    key(k, st)
    rows.append(row(name, st))
table(rows, 'paper/tables/main.tex', midrules=(3,))

# ----------------------------------------------------------------------------- sample-efficiency (M sweep)
sweep = []
for arm, label in [('ol_indep', 'Open loop, independent noise'), ('ol_crn3', 'Open loop, coupled noise'),
                   ('fbstep_indep', 'Feedback, independent noise'), ('fbstep_crn3', 'Feedback, coupled noise')]:
    cells = []
    for M, pat in [(4, f'results/d9/{arm}_M4_m[0-9].json'), (8, f'results/d9/{arm}_m[0-1].json'), (16, f'results/d9/{arm}_M16_m[0-9].json')]:
        fs = sorted(glob.glob(pat))
        if not fs:
            cells.append('--')
            continue
        n, ks, kf, kt, _ = from_episode_lists(fs)
        cells.append(f'{ks / n:.3f}/{kf / n:.3f}/{kt / n:.3f} ({n}' + (f', {len(fs)} model seed' if len(fs) < 2 else '') + ')')
        NUM[f'sweep_{arm}_M{M}'] = dict(n=n, success=ks / n, fall=kf / n, timeout=kt / n, nseeds=len(fs))
    sweep.append(f'{label} & ' + ' & '.join(cells) + r' \\')
open('paper/tables/sweep.tex', 'w').write('\n'.join([
    r'\begin{tabular}{lccc}', r'\toprule', r'Planner & $M=4$ & $M=8$ & $M=16$ \\', r'\midrule'] + sweep + [r'\bottomrule', r'\end{tabular}']) + '\n')


# ----------------------------------------------------------------------------- distribution shift
def phase_counts(files, phase):
    n = ks = kf = kt = 0
    for f in files:
        d = json.load(open(f))
        p = d['phases'].get(phase)
        if p:
            n += p['done']
            ks += p['success']
            kf += p['fall']
            kt += p['timeout']
    return n, ks, kf, kt


shift_rows = []
SH = [
    ('Deterministic predictor, mean rollout', 'suite_hi/shift/det_mean_s*.json'),
    ('Energy-score predictor, particles', 'suite_hi/shift/es_exp_s*.json'),
    ('\\quad CVaR$_{0.5}$ scoring', 'suite_hi/shift/es_cvar50_s*.json'),
    ('\\quad failure-rate controller (target 0.05)', 'suite_hi/shift/es_ctrl05_s*.json'),
    ('\\quad scalar spread recalibration (online)', 'suite_hi/adapt/es_exp_adapt_s*.json'),
    ('\\quad per-dimension spread recalibration', 'suite_hi/adapt2/es_exp_adaptvec_s*.json'),
    ('\\quad online fine-tuning of the predictor', 'suite_hi/tta/es_es_tta_s*.json'),
]
for name, pat in SH:
    fs = sorted(glob.glob(pat))
    if not fs:
        continue
    b = phase_counts(fs, 'before')
    a = phase_counts(fs, 'after')
    if a[0] == 0:
        continue
    NUM['shift_' + re.sub(r'\W+', '_', name)] = dict(before=dict(n=b[0], success=b[1] / b[0], fall=b[2] / b[0]), after=dict(n=a[0], success=a[1] / a[0], fall=a[2] / a[0], timeout=a[3] / a[0]))
    shift_rows.append(f'{name} & {b[0]} & {b[1] / b[0]:.3f} & {b[2] / b[0]:.3f} & {a[0]} & {a[1] / a[0]:.3f} & {a[2] / a[0]:.3f} & {a[3] / a[0]:.3f} \\\\')
open('paper/tables/shift.tex', 'w').write('\n'.join([
    r'\begin{tabular}{lrcc|rccc}', r'\toprule', r' & \multicolumn{3}{c}{Before shift} & \multicolumn{4}{c}{After shift} \\',
    r'Planner & Eps & Succ. & Fall & Eps & Succ. & Fall & Timeout \\', r'\midrule'] + shift_rows + [r'\bottomrule', r'\end{tabular}']) + '\n')


# ----------------------------------------------------------------------------- true-simulator reference planners
orows = []
for planner, plab in [('mean', 'Mean-dynamics planning'), ('exp', 'Particle-averaged planning')]:
    for cfg, clab in [('N32M8H10', '$N{=}32$, $M{=}8$, $H{=}10$'), ('N64M12H12', '$N{=}64$, $M{=}12$, $H{=}12$')]:
        if planner == 'mean' and cfg != 'N32M8H10':
            continue
        cells = []
        for w in ('09', '13'):
            f = f'results/oracle/{planner}_w{w}_{cfg}.json'
            if not os.path.exists(f):
                cells.append('--')
                continue
            d = json.load(open(f))
            e = d['episodes']
            cells.append(f'{d["success"]:.3f}/{d["fall"]:.3f}/{d["timeout"]:.3f} ({e})')
            NUM[f'oracle_{planner}_w{w}_{cfg}'] = dict(n=e, success=d['success'], fall=d['fall'], timeout=d['timeout'])
        orows.append(f'{plab} & {clab} & {cells[0]} & {cells[1]} \\\\')
open('paper/tables/oracle.tex', 'w').write('\n'.join([
    r'\begin{tabular}{llcc}', r'\toprule', r'Planner & Compute & Wind 0.09 & Wind 0.13 \\', r'\midrule'] + orows + [r'\bottomrule', r'\end{tabular}']) + '\n')

json.dump(NUM, open('paper/tables/numbers.json', 'w'), indent=1)

# ---- LaTeX macros for the prose (letters only in macro names)
def mname(k):
    return ''.join(ch for ch in k.title() if ch.isalpha()) if False else re.sub(r'[^A-Za-z]', '', {'det': 'det', 'ens3': 'enst', 'gauss': 'gauss', 'es_ol': 'esol', 'es_ol_crn': 'escrn', 'es_fb': 'esfb', 'es_fb_crn': 'esfbcrn'}.get(k, k))
mac = []
for k in ['det', 'ens3', 'gauss', 'es_ol', 'es_ol_crn', 'es_fb', 'es_fb_crn']:
    if k in NUM:
        d = NUM[k]
        m = mname(k)
        mac += [f'\\newcommand{{\\{m}N}}{{{d["n"]}}}', f'\\newcommand{{\\{m}Succ}}{{{d["success"]:.3f}}}',
                f'\\newcommand{{\\{m}Fall}}{{{d["fall"]:.3f}}}', f'\\newcommand{{\\{m}Time}}{{{d["timeout"]:.3f}}}']
open('paper/tables/macros.tex', 'w').write('\n'.join(mac) + '\n')
print(open('paper/tables/main.tex').read())
print(open('paper/tables/sweep.tex').read())
print(open('paper/tables/shift.tex').read())
