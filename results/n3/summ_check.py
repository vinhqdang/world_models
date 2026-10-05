import json,sys
for f in sys.argv[1:]:
    d=json.load(open(f)); print('==',f,'NS',d['NS'],'MT',d['MT'],'mean true fail prob',round(d['mean_true_fail_prob'],3),'particle-noise sd',round(d['mean_particle_noise_sd'],3))
    print('  N | optimism (pnoise+mbias) | regret(J8) regret(J64) pess1 pess2 | q_pt q_sel q_uni | cov_pt cov_sel cov_uni | cov_uni_all')
    for r in d['rows']:
        print('%3d | %5.2f (%5.2f%+5.2f) | %.3f %.3f %.3f %.3f | %.2f %.2f %.2f | %.2f %.2f %.2f | %.2f'%(r['N'],r['optimism'],r['particle_noise_part'],r['model_bias_part'],r['regret_vs_best_in_pool'],r['regret_J64_select'],r['regret_pess_beta1.0'],r['regret_pess_beta2.0'],r['q_pointwise'],r['q_selected'],r['q_uniform'],r['cov_pointwise'],r['cov_selected'],r['cov_uniform'],r['cov_uniform_all']))
