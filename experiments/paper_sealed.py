"""Sealed cell of the information-limited design (PLAN.md section 3, item 7). The cell is drawn AFTER the protocol commit:

    python3 experiments/paper_sealed.py PROTOCOL_COMMIT_HASH [--out results/paper/sealed_cell.json]

An RNG is seeded with the first 16 hexadecimal digits of the commit hash (nothing else enters the seed). It draws
  environment  uniformly from {E1o, E2o, E3o}
  pi           log-uniform on [0.08, 0.60] (rounded to 3 decimals)
  amplitude    a multiplicative factor U[0.85, 1.15] applied to the calibrated per-axis command budget sigma interpolated (in log pi)
               between the calibrated gated cells of that environment (pi = 0.05, 0.1, 0.2, 0.5, 1.0; results/cwk/CALIBRATION.json),
               clipped to [0.25, 0.95]
and writes the configuration (including the hash, the seed and every draw) to the output file. r = d + 1 (offset + one gain per
actuator), iid, no calibration or inspection of the cell's outcomes is allowed after the draw: the oracle R^2 of the realised cell
is reported as it comes. No data are produced by this script. The hash must be recorded (committed) before any data of the sealed
cell are generated."""
import sys, json, os, argparse
sys.path.insert(0, "/home/user/world_models")
import numpy as np

CAL = "/home/user/world_models/results/cwk/CALIBRATION.json"
PIS = [0.05, 0.1, 0.2, 0.5, 1.0]


def draw(commit_hash, cal_path=CAL):
    h = commit_hash.strip().lower()
    assert len(h) >= 16 and all(c in "0123456789abcdef" for c in h), "give a hexadecimal commit hash"
    seed = int(h[:16], 16)
    rng = np.random.default_rng(seed)
    env = ["E1o", "E2o", "E3o"][int(rng.integers(0, 3))]
    pi = float(np.round(np.exp(rng.uniform(np.log(0.08), np.log(0.60))), 3))
    amp = float(np.round(rng.uniform(0.85, 1.15), 4))
    cal = json.load(open(cal_path))
    d = {"E1o": 2, "E2o": 2, "E3o": 3}[env]
    keys = [f"info_{env}_gated_p{int(round(p * 100))}_r{d + 1}" for p in PIS]
    sig = [cal[k]["sigma"] for k in keys]
    base = float(np.interp(np.log(pi), np.log(PIS), sig))
    sigma = float(np.round(np.clip(base * amp, 0.25, 0.95), 4))
    return dict(commit_hash=h, seed=seed, env=env, pi=pi, amplitude_factor=amp, sigma_calibrated_interp=base, sigma=sigma,
                gain=float(0.98 * (1 - sigma) / {"E1o": 0.45, "E2o": 0.35, "E3o": 0.48}[env]), r=d + 1, d=d, iid=1.0,
                seed_offset=3_000_000, calibration_keys=keys, calibration_sigmas=sig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("commit_hash")
    ap.add_argument("--out", default="/home/user/world_models/results/paper/sealed_cell.json")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: the sealed cell is drawn once")
    cfg = draw(a.commit_hash)
    json.dump(cfg, open(a.out, "w"), indent=1)
    print(json.dumps(cfg, indent=1))
