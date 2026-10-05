"""Turn results/d4/d4_verify.json into the LaTeX table embedded in paper/theory.tex (replaces the marker block)."""
import json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = json.load(open(os.path.join(root, 'results', 'd4', 'd4_verify.json')))
desc = {
 'T1a_identity': ('T1', 'bias--variance identity (rel.\\ MC error)'),
 'T1b_mse_to_mean': ('T1', 'MSE fit converges to the mean (sup error, $n=10^6$)'),
 'T1c_identity': ('T1', 'regret identity, Thm~\\ref{thm:regret}(a)'),
 'T1c_bound_violations': ('T1', 'violations of $R\\le\\Delta_v$ (count, 12 instances)'),
 'T1c_MC_regret': ('T1', 'MC regret vs exact (abs.)'),
 'T1d_tightness': ('T1', 'tightness example $R=V-\\eta$ (rel.)'),
 'T1e_fail_penalty': ('T1', 'failure-penalised regret, $\\kappa\\le500$ (rel.)'),
 'T1f_cvar_closed_form': ('T1', '$c_\\alpha$ closed form vs MC tail mean (rel.)'),
 'T1f_cvar_regret': ('T1', 'CVaR regret (rel.)'),
 'T1f_cvar_bounds': ('T1', '$\\EE C\\le\\CVaR_\\alpha\\le\\EE C/(1-\\alpha)$ (violations)'),
 'T2a_select_prob': ('T2', 'selection probability, 40 grid cells (abs.)'),
 'T2a_monotone_in_Nr': ('T2', 'monotone in $N_r$ (violations)'),
 'T2b_excess_cost': ('T2', 'excess true cost (abs.)'),
 'T2b_winner_curse': ('T2', '$\\EE\\min\\hat J$ formula (abs.)'),
 'T2c_hoeffding_violations': ('T2', 'Hoeffding bound (violations)'),
 'T2d_verification_fallback': ('T2', 'fresh verification, $1-(1-F\')^K$ (abs.)'),
 'T2d_verification_pooled': ('T2', 'pooled variant, Remark~\\ref{rem:fallback} (abs.)'),
 'T2e_fresh_unbiased': ('T2', 'fresh estimate unbiased given selection (abs.)'),
 'T2f_shrink_equalM': ('T2', 'argmin changed by shrinkage, equal $M$ (fraction)'),
 'T2f_shrink_unequalM_example': ('T2', 'unequal-$M$ flip exhibited'),
 'T3a_es_closed_form': ('T3', 'ES of a Gaussian, closed form (abs.)'),
 'T3b_propriety_grid': ('T3', 'ES minimised at truth, $(m,s)$ grid (dist.)'),
 'T3c_energy_distance_identity': ('T3', '$\\EE\\ES(P,Y)-\\EE\\ES(Q,Y)=\\frac12\\mathcal E$ (abs.)'),
 'T3d_point_mass_median': ('T3', 'point-mass ES gives median (abs.)'),
 'T3e_crn_variance': ('T3', 'CRN variance $4s^2(m_1-\\epsilon m_2)^2$ (rel.)'),
 'T3f_consistency': ('T3', 'dist.\\ to $\\Theta_\\star$ at largest $n$ (median)'),
 'T3g_vstat_improper': ('T3', 'V-statistic minimiser $s_m$ (abs.)'),
 'T4a_variance_growth': ('T4', 'variances in Prop.~\\ref{prop:var} (rel.)'),
 'T4b_LQG_ratio_H': ('T4', 'values $H\\sigma^2\\rho$ vs $\\sigma^2\\rho$ (rel.)'),
 'T4b_cvar_monotone_in_offset': ('T4', 'CVaR increasing in $|m|$ (violations)'),
 'T4b_entropic_H1': ('T4', 'entropic risk, $H=1$ (rel.)'),
 'T4c_barrier_MC_vs_grid': ('T4', 'barrier DP / OL optimum vs MC (std.\\ errors)'),
 'T4c_barrier_gap_positive': ('T4', 'min gap $V_{\\rm OL}-V_{\\rm CL}$, $H=2..5$ (must be $>0$)'),
 'T4c_ystar_below_B': ('T4', 'hypotheses of Thm~\\ref{thm:barrier} hold'),
 'T4c_marginal_exceedance': ('T4', 'exceedance probabilities, Prop.~\\ref{prop:var}(c) (abs.)'),
 'T4c_firstpassage_crossover': ('T4', 'FB first passage $\\le$ OL for $t\\le20$ (see Remark~\\ref{rem:firstpassage})'),
 'T5a_laplace_tails_MC': ('T5', 'Laplace tails (rel.)'),
 'T5b_decision_counterexample': ('T5', 'scales fixing both decisions (count, must be 0)'),
 'T5c_floor_twopoint_ge_quarter': ('T5', 'floor $\\ge1/4$ (shortfall)'),
}
rows = []
for k, (grp, d) in desc.items():
    if k not in R: continue
    v = R[k]
    ok = v['ok']
    mark = 'pass' if ok else ('FAIL' if ok is not None else 'n/a')
    if k in ('T4c_barrier_gap_positive',):
        val = '%.3f' % v['max_err']
    else:
        val = '%.1e' % v['max_err']
        val = val.replace('e-0', '\\!\\times\\!10^{-').replace('e+0', '\\!\\times\\!10^{').replace('e-', '\\!\\times\\!10^{-').replace('e+', '\\!\\times\\!10^{')
        val += '}'
        if v['max_err'] == 0: val = '0'
    rows.append(f'{grp} & {d} & ${val}$ & {mark}\\\\')
tab = ('\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{@{}llrl@{}}\n\\hline\nGroup & Check & Max.\\ error & Result\\\\\n\\hline\n'
       + '\n'.join(rows) + '\n\\hline\n\\end{tabular}\n\\caption{Largest error per checked claim (\\texttt{results/d4/d4\\_verify.json}). '
       'Monte-Carlo entries are sampling errors ($S=2\\times10^6$ draws unless stated). Tolerances are set at about three standard errors.}\n'
       '\\label{tab:verif}\n\\end{table}')
p = os.path.join(root, 'paper', 'theory.tex')
s = open(p).read()
begin, end = '%%VERIF_TABLE_BEGIN\n', '\n%%VERIF_TABLE_END'
if begin in s:
    s = re.sub(re.escape(begin) + '.*?' + re.escape(end), lambda m: begin + tab + end, s, flags=re.S)
else:
    s = s.replace('\\input{theory_verif_table}', begin + tab + end)
open(p, 'w').write(s)
print(tab)
