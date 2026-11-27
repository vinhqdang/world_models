"""R1: 5-seed confirmation of the Stage 2 findings on the nine cached Stage 2 cells with the full paper arm set.
usage: python3 experiments/paper_r1.py [CELL ...] [--seeds 5] [--smoke]       (reports: python3 experiments/paper_report.py r1)
Arms: rr_learned and trained (always), hh_global, hh_binned, hh_binned_nug, hh_nug_ref*, rc_eb_gmm, oracle_kernel_lm*, oracle_quad*,
pck_pol (the merged algorithm, for the tie claim); hh_basis_{x,abs,x_x3,cubic} (collinearity) on the saturating cell only;
deterministic: oracle*, rr_exact*, oracle_kernel*, oracle_quad_ex*. Cached Stage 2 data only (no new simulation)."""
from paper_common import *

ARMS = ["hh_global", "hh_binned", "hh_binned_nug", "hh_nug_ref", "oracle_kernel_lm", "oracle_quad", "rc_eb_gmm", "pck_pol"]
BASIS = ["hh_basis_x", "hh_basis_abs", "hh_basis_x_x3", "hh_basis_cubic"]
TAG = "r1"


def arms_for(name):
    return ARMS + (BASIS if name == "E1_nu50_sat_iid50" else [])


if __name__ == "__main__":
    a = cli(sys.argv[1:])
    names = a.cells or STAGE2_NINE
    if a.smoke:
        w, cell, tmp = smoke_cell(names[0], ["hh_binned_nug", "hh_nug_ref", "oracle_kernel_lm", "oracle_quad", "rc_eb_gmm", "hh_basis_abs"], TAG)
        print("R1 smoke ok", tmp)
        sys.exit(0)
    log = lambda *x, **k: print(*x, flush=True)
    for n in names:
        run_cell(n, arms_for(n), range(a.seeds), TAG, log=log)
