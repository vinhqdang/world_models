"""Summarise n1_diag JSONs: per arm J (cost of returned plan on 1024 fresh draws), failure prob, optimism; paired state-bootstrap
CI of J difference to a reference arm.  usage: n1_diag_summ.py ref_arm file1.json [file2.json ...]  (files = states pooled)"""
import json, sys
import numpy as np
ref = sys.argv[1]; files = sys.argv[2:]
rs = [json.load(open(f)) for f in files]
arms = list(rs[0]['arms'])
J = {a: np.concatenate([r['arms'][a]['J'] for r in rs if a in r['arms']]) for a in arms}
Pf = {a: np.concatenate([r['arms'][a]['pfail'] for r in rs if a in r['arms']]) for a in arms}
opt = {a: np.mean([r['arms'][a]['optimism_vs_Abest'] for r in rs if a in r['arms']]) for a in arms}
ch = {a: [r['arms'][a]['frac_choice_mean_plan'] for r in rs if a in r['arms']] for a in arms}
rng = np.random.default_rng(1)
n = len(J[ref])
print(f'states pooled: {n}  (files: {len(files)})')
print('| arm | J (true-model cost of returned plan) | diff vs ' + ref + ' [95% CI, paired over states] | fail prob | diff [95% CI] | optimism J_fresh - in-sample(best elite) | frac. mean-plan chosen |')
print('|---|---|---|---|---|---|---|')
for a in arms:
    if len(J[a]) != n: continue
    idx = rng.integers(0, n, (4000, n))
    dJ = (J[a] - J[ref])[idx].mean(1); dP = (Pf[a] - Pf[ref])[idx].mean(1)
    c = ch[a][0]
    print(f'| {a} | {J[a].mean():.4f} | {np.mean(J[a]-J[ref]):+.4f} [{np.percentile(dJ,2.5):+.4f}, {np.percentile(dJ,97.5):+.4f}] | {Pf[a].mean():.4f} | '
          f'{np.mean(Pf[a]-Pf[ref]):+.4f} [{np.percentile(dP,2.5):+.4f}, {np.percentile(dP,97.5):+.4f}] | {opt[a]:+.3f} | {"-" if c is None else f"{np.mean([x for x in ch[a]]):.3f}"} |')
