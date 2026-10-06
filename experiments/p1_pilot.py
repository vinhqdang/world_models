"""P1 pilot driver. Usage:
  python experiments/p1_pilot.py --seeds 900 901 902 --sigma 0.15 --methods mle idm --lam 1 --out results/p1/dev_x.json
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selwm import p1_env as E

torch.set_num_threads(1)


def run_seed(seed, sigma, N, methods, lams, epochs=40, n_prob=150, H=6, wd=1e-4):
    env = E.Env(seed)
    S, A, S2 = env.collect(N, sigma, seed=seed * 7 + 1)
    Sv, Av, S2v = env.collect(2000, sigma, seed=seed * 7 + 2)
    ehat = E.make_ehat(S, A, seed)
    r2 = 1 - float(((A - ehat(S)) ** 2).mean() / A.var(0).mean())
    s0, goal = E.make_problems(env, Sv, n_prob, H, seed=seed * 7 + 3)
    # oracle planner on true dynamics (same CEM, same noise seed)
    ostep = lambda s, a: env.step(s, a)
    oplan = E.cem_plan(ostep, s0, goal, H=H, seed=seed * 7 + 4)
    c_or = E.true_cost(env, s0, goal, oplan)
    c_zero = E.true_cost(env, s0, goal, torch.zeros(n_prob, H, E.DA))
    rows = []
    for m in methods:
        t0 = time.time()
        if m == "oracle_int":  # reference only: interventional data (uniform actions), same N
            Si, Ai, S2i = env.collect(N, sigma, seed=seed * 7 + 5, uniform=True)
            model = E.train("mle", Si, Ai, S2i, seed, epochs=epochs, wd=wd)
        elif m.startswith("mix"):  # budget-matched: fraction rho of the N transitions use uniform actions
            rho = float(m[3:])
            k = int(rho * N)
            Si, Ai, S2i = env.collect(k + 20, sigma, seed=seed * 7 + 5, uniform=True)
            Sm = torch.cat([S[: N - k], Si[:k]]); Am = torch.cat([A[: N - k], Ai[:k]]); S2m = torch.cat([S2[: N - k], S2i[:k]])
            model = E.train("mle", Sm, Am, S2m, seed, epochs=epochs, wd=wd)
        else:
            model = E.train(m, S, A, S2, seed, lam=lams.get(m, 1.0), epochs=epochs, wd=wd, ehat=ehat)
        mse = float(((model(Sv, Av) - S2v) ** 2).mean())
        am = E.action_metrics(env, model, Sv, ehat, seed=seed * 7 + 6)
        c, _ = E.planning_eval(env, model, s0, goal, H=H, seed=seed * 7 + 4)
        reg = float((c - c_or).mean())
        nreg = reg / float(c_zero.mean())
        rows.append(dict(seed=seed, sigma=sigma, N=N, method=m, mse=mse, regret=reg, nregret=nreg,
                         cost=float(c.mean()), c_oracle=float(c_or.mean()), c_zero=float(c_zero.mean()),
                         r2_a_given_s=r2, secs=time.time() - t0, **am))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--N", type=int, default=6000)
    ap.add_argument("--methods", nargs="+", default=["mle"])
    ap.add_argument("--lam", type=float, nargs="*", default=[])  # one per method (aux weight)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lams = {m: (a.lam[i] if i < len(a.lam) else 1.0) for i, m in enumerate(a.methods)}
    allrows = []
    for sd in a.seeds:
        rows = run_seed(sd, a.sigma, a.N, a.methods, lams, a.epochs, wd=a.wd)
        for r in rows:
            r["lam"] = lams.get(r["method"], None)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        allrows += rows
        json.dump(allrows, open(a.out, "w"))
