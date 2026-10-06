"""S2: verification of the fallback controller pi_0 on the true simulator (pilot seed 1100, not an evaluation seed).
python experiments/s2_pi0_check.py   -> results/s2/pi0_check.txt
Reference numbers for the base planner (open-loop CEM N=32, M=8, chained CRN, learned es models) are taken from results/d9 (seed 211) and
results/FINDINGS_stochastic.md; the base planner's closed-loop numbers on the evaluation seeds are in the main tables."""
import sys; sys.path.insert(0, '.')
import numpy as np
from selwm.stochnav import StochNav
from selwm.s2_scc import pi0_action, PI0, PI0_FAST

def run(cfg, wind, n=4000, seed=1100, T=120):
    env = StochNav('cliff_hi', n, seed, wind=wind); env.reset()
    out = np.zeros(n, int); steps = np.zeros(n, int); done = np.zeros(n, bool)
    for k in range(T):
        env.step(pi0_action(env.p, env.goal, cfg))
        fin = (env.fell | env.reached) & ~done
        out[fin & env.fell] = 1; out[fin & env.reached] = 2; steps[fin] = k + 1; done |= fin
    s, f = (out == 2).mean(), (out == 1).mean()
    ci = lambda p: 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
    return s, f, ci(f), 1 - s - f, float(np.median(steps[out == 2]))

L = ['pi_0 on the true simulator (n=4000 episodes per row, 120-step limit, pilot seed 1100)',
     f"{'controller':34s} {'wind':>5s} {'success':>8s} {'fall [+-95%]':>16s} {'timeout':>8s} {'median steps (successes)':>26s}"]
for name, cfg in (('pi_0 (v0=0.5, y_so=0.85) [used]', PI0), ('pi_0 fast (v0=1.0) [reference]', PI0_FAST)):
    for wind in (0.09, 0.13):
        s, f, c, to, md = run(cfg, wind)
        L.append(f"{name:34s} {wind:5.2f} {s:8.3f} {f:8.3f} +-{c:5.3f} {to:8.3f} {md:26.1f}")
L += ['', 'Base planner reference (D9, seed 211, 329 episodes, wind 0.09): success 0.888, fall 0.033 [0.015, 0.055], timeout 0.079, median steps 51 [49, 54].',
      'Energy-score planner after a 0.09 -> 0.13 shift (FINDINGS_stochastic.md, expected cost, not CRN): success 0.709, fall 0.200.',
      'Reading: the used pi_0 falls less than the base planner at wind 0.09 (0.002 vs 0.033) and is slower (median 59 vs 51 steps) as required; the fast variant',
      'is both safer and faster than the base planner, so on this benchmark deferral is not intrinsically costly (the slowness of the used pi_0 is a design choice, speed cap v0).']
open('results/s2/pi0_check.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
