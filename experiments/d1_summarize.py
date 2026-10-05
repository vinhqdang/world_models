"""Summarise results/d1/*_s{seed}.json: per-config before/after-shift success/fall/timeout (pooled over seeds, with episode counts) and gate timing."""
import glob, json, os, re, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else 'results/d1'
pat = sys.argv[2] if len(sys.argv) > 2 else '*'
rows = {}
for f in sorted(glob.glob(f'{root}/{pat}_s[0-9].json')):
    m = re.match(r'(.+)_s(\d+)\.json', os.path.basename(f))
    rows.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
print(f'{"config":22s} {"seeds":6s} | {"BEFORE n":>8s} {"succ":>5s} {"fall":>5s} {"to":>5s} | {"AFTER n":>7s} {"succ":>5s} {"fall":>5s} {"to":>5s} | gate first-on (abs / after-shift), pre-shift on-steps')
for name, sd in rows.items():
    out = []
    for ph in ('before', 'after'):
        n = sum(sd[s]['phases'].get(ph, {}).get('done', 0) for s in sd)
        c = {k: sum(sd[s]['phases'].get(ph, {}).get(k, 0) for s in sd) for k in ('success', 'fall', 'timeout')}
        out.append((n, c))
    g = [(sd[s].get('gate_first_on'), sd[s].get('gate_first_on_after_shift'), sd[s].get('gate_on_steps_before')) for s in sorted(sd)]
    fmt = lambda n, c: f'{n:8d} {c["success"]/max(n,1):5.3f} {c["fall"]/max(n,1):5.3f} {c["timeout"]/max(n,1):5.3f}'
    gs = ' '.join(f'[s{s}: {a}/{b}, pre={p}]' for s, (a, b, p) in zip(sorted(sd), g)) if any(x[0] is not None or x[2] is not None for x in g) else '-'
    print(f'{name:22s} {",".join(map(str, sorted(sd))):6s} | {fmt(*out[0])} | {fmt(*out[1])} | {gs}')
