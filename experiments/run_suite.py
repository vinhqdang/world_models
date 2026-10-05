"""Train state-input JEPA world models and evaluate planner configurations on a stochastic-navigation variant.

python experiments/run_suite.py --variant cliff --seeds 3 --root suite_cliff
Everything is resumable: existing checkpoints / result files are skipped.
"""
import argparse, itertools, json, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='cliff'); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--root', default='suite'); ap.add_argument('--steps', type=int, default=10000)
ap.add_argument('--dim', type=int, default=3); ap.add_argument('--obs', default='fixed'); ap.add_argument('--E', type=int, default=64)
ap.add_argument('--total_steps', type=int, default=360); ap.add_argument('--N', type=int, default=64)
ap.add_argument('--M', type=int, default=8); ap.add_argument('--H', type=int, default=10)
ap.add_argument('--only', default=''); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--iters', type=int, default=3); ap.add_argument('--device', default='cuda'); ap.add_argument('--extra', default='')
args = ap.parse_args()
os.makedirs(f'{args.root}/ckpt', exist_ok=True); os.makedirs(f'{args.root}/res', exist_ok=True)


def sh(cmd):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print('FAILED', cmd, r.stderr[-500:], flush=True)
    return r


# ---------------------------------------------------------------- models
for kind, seed in itertools.product(['det', 'gauss', 'es'], range(args.seeds)):
    f = f'{args.root}/ckpt/{kind}_s{seed}.pt'
    if not os.path.exists(f):
        sh(f'python3 -u experiments/train_wm.py --kind {kind} --obs {args.obs} --dim {args.dim} --steps {args.steps} --n_data 300000 '
           f'--variant {args.variant} --seed {seed} --device {args.device} --out {f} > {args.root}/ckpt/log_{kind}_s{seed}.txt 2>&1')
        print('trained', f, flush=True)

# ---------------------------------------------------------------- planner configurations
C = lambda kind, seed: f'{args.root}/ckpt/{kind}_s{seed}.pt'
cfgs = {   # name -> (ckpt-fn(seed), flags)
    'det_mean':       (lambda s: C('det', s), '--planner mean'),
    'gauss_mean':     (lambda s: C('gauss', s), '--planner mean'),
    'gauss_exp':      (lambda s: C('gauss', s), '--planner risk --lam 0'),
    'gauss_cvar':     (lambda s: C('gauss', s), '--planner risk --lam 1.0'),
    'es_exp':         (lambda s: C('es', s), '--planner risk --lam 0'),
    'es_cvar50':      (lambda s: C('es', s), '--planner risk --lam 0.5'),
    'es_cvar100':     (lambda s: C('es', s), '--planner risk --lam 1.0'),
    'es_exp_shrink':  (lambda s: C('es', s), '--planner risk --lam 0 --shrink 1'),
    'es_cvar50_shrink': (lambda s: C('es', s), '--planner risk --lam 0.5 --shrink 1'),
    'es_cvar50_race': (lambda s: C('es', s), '--planner risk --lam 0.5 --shrink 1 --race 1'),
    'es_cvar50_crn':  (lambda s: C('es', s), '--planner risk --lam 0.5 --shrink 1 --crn 1'),
    'ens3_exp':       (lambda s: ','.join(C('det', (s + j) % args.seeds) for j in range(3)), '--planner ensemble --M 9'),
}
common = f'--variant {args.variant} --E {args.E} --N {args.N} --M {args.M} --H {args.H} --iters {args.iters} --total_steps {args.total_steps} --device {args.device} {args.extra}'
jobs = []
for name, (ck, flags) in cfgs.items():
    if args.only and name not in args.only.split(','):
        continue
    for s in range(args.seeds):
        out = f'{args.root}/res/{name}_s{s}.json'
        if not os.path.exists(out):
            jobs.append((name, s, out, f'python3 experiments/eval_stream.py --ckpt {ck(s)} {flags} {common} --seed {s} --out {out} > /dev/null 2>&1'))

from concurrent.futures import ThreadPoolExecutor
def work(j):
    name, s, out, cmd = j
    sh(cmd)
    r = json.load(open(out)) if os.path.exists(out) else {}
    print(f'{name:18s} seed {s}: success {r.get("success", float("nan")):.3f} fall {r.get("fall", float("nan")):.3f} '
          f'timeout {r.get("timeout", float("nan")):.3f} episodes {r.get("episodes")}', flush=True)
with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(work, jobs))
open(f'{args.root}/DONE', 'w').write('done')
