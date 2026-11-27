"""R3: upstream cells (the authors' own configurations), no-harm check of the HH / EB arms against rr_learned.
usage: python3 experiments/paper_r3.py [CELL ...] [--seeds 5] [--smoke]
Cells: hand_l0 (the authors' 'hand'), cup, fetch, hand_iid50, cup_iid50, fetch_iid50, hand_load, hand_affine50, hand_affine100.
Arms: hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, rc_npmle, rc_flow, pck_pol (+ rr_learned, trained)."""
from paper_common import *

UP = ["hand_l0", "cup", "fetch", "hand_iid50", "cup_iid50", "fetch_iid50", "hand_load", "hand_affine50", "hand_affine100"]
ARMS = ["hh_global", "hh_binned", "hh_binned_nug", "rc_eb_gmm", "rc_npmle", "rc_flow", "pck_pol", "oracle_kernel_lm"]
TAG = "r3"
UP_BY_NAME = {"hand_l0": "UP_hand", "cup": "UP_cup", "fetch": "UP_fetch", "hand_iid50": "UP_hand_iid50", "cup_iid50": "UP_cup_iid50",
              "fetch_iid50": "UP_fetch_iid50", "hand_load": "UP_hand_load", "hand_affine50": "UP_hand_affine50",
              "hand_affine100": "UP_hand_affine100"}

if __name__ == "__main__":
    a = cli(sys.argv[1:])
    names = a.cells or UP
    if a.smoke:
        w, cell, tmp = smoke_cell(names[0] if a.cells else "hand_l0", ["hh_binned_nug", "rc_eb_gmm"], TAG)
        print("R3 smoke ok", tmp)
        sys.exit(0)
    log = lambda *x, **k: print(*x, flush=True)
    for n in names:
        run_cell(n, ARMS, range(a.seeds), TAG, log=log)
