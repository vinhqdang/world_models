"""Figures of the manuscript, generated from the result files and the simulator (no hand-typed values).

python paper/make_figures.py  (from the repository root; needs paper/tables/numbers.json from make_tables.py)
Writes paper/figs/*.pdf. Palette: fixed categorical slots (blue, orange, aqua, violet), hatch textures for print.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

sys.path.insert(0, '.')
os.makedirs('paper/figs', exist_ok=True)
NUM = json.load(open('paper/tables/numbers.json'))

BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e3e3e0'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'hatch.linewidth': 0.6})

# ------------------------------------------------------------------ Fig. 1: benchmark and example paths
def fig_benchmark():
    import torch
    from selwm.stochnav import StochNav
    from selwm.risk_plan import cem, OracleModel
    torch.set_num_threads(1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, (planner, title) in zip(axes, [('mean', 'Deterministic (mean) planning'), ('expected', 'Particle-averaged planning')]):
        env = StochNav('cliff_hi', 10, 3)
        env.reset()
        gen = torch.Generator().manual_seed(1)
        model = OracleModel('cliff_hi', deterministic=(planner == 'mean'), stage_w=1.0)
        paths = [[p.copy()] for p in env.p]
        fell_at = {}
        warm = None
        for t in range(90):
            p = torch.tensor(env.p, dtype=torch.float32)
            g = torch.tensor(env.goal, dtype=torch.float32)
            plan = cem(model, p, g, 12, 64, 12, iters=3, risk='expected', gen=gen, warm=warm)
            warm = torch.cat([plan[:, 1:], torch.zeros_like(plan[:, :1])], 1)
            prev_alive = ~(env.fell | env.reached)
            env.step(plan[:, 0].numpy())
            for i in range(len(paths)):
                if prev_alive[i]:
                    paths[i].append(env.p[i].copy())
                    if env.fell[i] and i not in fell_at:
                        fell_at[i] = len(paths[i]) - 1
        ax.add_patch(plt.Rectangle((0.25, 0), 0.5, 0.30, fc='#cfd3dc', ec=MUTED, hatch='///', lw=0.8))
        ax.text(0.5, 0.15, 'pit (absorbing)', ha='center', va='center', color=INK, fontsize=7)
        for i, pth in enumerate(paths):
            a = np.array(pth)
            ax.plot(a[:, 0], a[:, 1], color=ORANGE if i in fell_at else BLUE, lw=0.9, alpha=0.85)
            if i in fell_at:
                ax.plot(a[-1, 0], a[-1, 1], 'x', color=ORANGE, ms=4)
            else:
                ax.plot(a[0, 0], a[0, 1], 'o', color=BLUE, ms=2.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks([0, 0.5, 1])
        ax.text(0.02, 0.76, f'{len(fell_at)}/10 fell', color=ORANGE, fontsize=7)
    axes[0].set_ylabel('y')
    for ax in axes:
        ax.set_xlabel('x')
    fig.savefig('paper/figs/benchmark.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2: outcomes per planner
def fig_outcomes():
    order = [('det', 'Deterministic, mean rollout'), ('ens3', 'Ensemble of 3 deterministic'), ('gauss', 'Gaussian, particles'),
             ('es_ol', 'Energy score, particles'), ('es_ol_crn', '  + coupled noise'), ('es_fb', '  + feedback'), ('es_fb_crn', '  + feedback + coupled noise')]
    rows = [(lab, NUM[k]) for k, lab in order if k in NUM]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d) in zip(y, rows):
        left = 0.0
        for name, color, hatch in [('success', BLUE, ''), ('fall', ORANGE, '///'), ('timeout', AQUA, '\\\\\\')]:
            w = d[name]
            ax.barh(yi, w, left=left, height=0.62, color=color, edgecolor='white', linewidth=1.2, hatch=hatch)
            if w >= 0.07:
                ax.text(left + w / 2, yi, f'{w:.2f}', ha='center', va='center', color='white', fontsize=7)
            left += w
        ax.text(1.01, yi, f'n={d["n"]}', va='center', color=MUTED, fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of episodes')
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=BLUE, label='success'), Patch(fc=ORANGE, hatch='///', ec='white', label='fall'), Patch(fc=AQUA, hatch='\\\\\\', ec='white', label='timeout')],
              ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), fontsize=7)
    fig.savefig('paper/figs/outcomes.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3: sample efficiency (small multiples, shared x)
def fig_sweep():
    arms = [('ol_indep', 'open loop', BLUE), ('ol_crn3', 'open loop + coupled noise', ORANGE), ('fbstep_indep', 'feedback', AQUA), ('fbstep_crn3', 'feedback + coupled noise', VIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharex=True)
    for ax, (metric, lab) in zip(axes, [('success', 'success'), ('fall', 'fall'), ('timeout', 'timeout')]):
        for arm, name, c in arms:
            pts = [(M, NUM[f'sweep_{arm}_M{M}']) for M in (4, 8, 16) if f'sweep_{arm}_M{M}' in NUM]
            full = [(M, d[metric]) for M, d in pts if d['nseeds'] >= 2]
            if full:
                ax.plot(*zip(*full), '-o', color=c, lw=1.4, ms=4, markeredgecolor='white', markeredgewidth=0.8, label=name)
            single = [(M, d[metric]) for M, d in pts if d['nseeds'] < 2]
            if single:                       # one model seed only: hollow marker, no connecting line
                ax.plot(*zip(*single), 'o', mfc='white', mec=c, mew=1.2, ms=4.5)
        ax.set_xscale('log', base=2)
        ax.set_xlim(3.3, 19.5)
        ax.set_xticks([4, 8, 16])
        ax.set_xticklabels(['4', '8', '16'])
        ax.set_xlabel('particles per candidate $M$')
        ax.set_title(lab, fontsize=8, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.38)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker='o', mfc='white', mec=MUTED, ls='', label='single model seed'))
    fig.legend(h, l + ['single model seed'], ncol=3, frameon=False, fontsize=6.8, loc='lower center', bbox_to_anchor=(0.5, -0.30))
    fig.savefig('paper/figs/sweep.pdf')
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 4: tail blindness amplified by selection (theory vs simulation)
def fig_tail():
    rng = np.random.default_rng(0)
    p, kappa, cs, cr = 0.1, 3.0, 0.2, 0.0
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    Nr = np.arange(1, 21)
    for M, c in [(4, BLUE), (8, ORANGE), (16, AQUA), (32, VIOLET)]:
        kstar = math.floor(M * (cs - cr) / kappa)
        F = binom.cdf(kstar, M, p)
        theory = 1 - (1 - F) ** Nr
        ax.plot(Nr, theory, color=c, lw=1.4, label=f'$M={M}$')
        mc = []
        for n in Nr[::3]:
            k = rng.binomial(M, p, size=(4000, n))
            mc.append(((cr + kappa * k / M).min(1) < cs).mean())
        ax.plot(Nr[::3], mc, 'o', color=c, ms=3.2, markeredgecolor='white', markeredgewidth=0.6)
    ax.set_xlabel('number of risky candidates $N_r$')
    ax.set_ylabel('P(select a risky candidate)')
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc='lower right', title='lines: Thm. 2; dots: simulation', title_fontsize=6.5)
    fig.savefig('paper/figs/tail.pdf')
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['benchmark', 'outcomes', 'sweep', 'tail']
    for w in which:
        {'benchmark': fig_benchmark, 'outcomes': fig_outcomes, 'sweep': fig_sweep, 'tail': fig_tail}[w]()
        print('wrote', w)
