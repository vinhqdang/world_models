"""R6: predictor-accuracy sensitivity. The iid100 Stage 2 cells with a mean predictor trained on 512 episodes only (the
kernel / EB arms are fitted on the same residuals of all 16384 episodes, out-of-sample for the predictor outside its 512).
usage: python3 experiments/paper_r6.py [CELL ...] [--seeds 5] [--smoke]       (report: paper_report.py r6)
Arms: hh_global, hh_binned, hh_binned_nug, rc_eb_gmm, oracle_kernel_lm* (+ rr_learned, trained)."""
from paper_common import *

CELLS = ["E1_nu50_gain_iid100", "E2_cup_iid100", "E3_fetch_iid100"]
ARMS = ["hh_global", "hh_binned", "hh_binned_nug", "rc_eb_gmm", "oracle_kernel_lm"]
TAG = "r6"

if __name__ == "__main__":
    a = cli(sys.argv[1:])
    if a.smoke:
        w, cell, tmp = smoke_cell(CELLS[0], ["hh_binned_nug", "rc_eb_gmm"], TAG, extra_cfg=dict(pred_episodes=100))
        print("R6 smoke ok", tmp)
        sys.exit(0)
    log = lambda *x, **k: print(*x, flush=True)
    for n in a.cells or CELLS:
        run_cell(n, ARMS, range(a.seeds), TAG, extra_cfg=dict(pred_episodes=512), log=log)
