"""Resumable job queue for phase 2 (no result logic). python3 experiments/paper_queue.py heavy|sim
Tasks are (name, command, prerequisite ready-files). A task is claimed by mkdir (atomic), logged to results/paper/logs/<name>.log and
marked by <name>.done; a claim without .done at start-up (crash / kill) is released. Per-seed results are cached by the runner, so a
re-run only continues missing seeds."""
import sys, os, time, subprocess, glob
sys.path.insert(0, "/home/user/world_models/experiments")
L = "/home/user/world_models/results/paper/logs"
CL = f"{L}/claims"
os.makedirs(CL, exist_ok=True)
PY = "python3 /home/user/world_models/experiments"
ENVS = ["E1o", "E2o", "E3o"]
S5 = ["P", "T", "G", "Piid50"]
SW = ["pi5", "pi10", "pi50", "pi100", "rd", "rd2", "Pdata1", "Pdata2"]
R2_ORDER = [f"{e}_{c}" for c in S5 for e in ENVS] + ["SEALED"] + [f"{e}_{c}" for c in SW for e in ENVS]
R3_CELLS = ["hand_l0", "cup", "fetch", "hand_iid50", "cup_iid50", "fetch_iid50", "hand_load", "hand_affine50", "hand_affine100"]
STAGE2 = ["E1_nu50_gain_iid50", "E1_nu80_gain_iid50", "E2_cup_iid50", "E3_fetch_iid50", "E1_nu50_sat_iid50", "E1_nu50_deadzone_iid50",
          "E1_nu50_gain_iid100", "E2_cup_iid100", "E3_fetch_iid100"]
rd = lambda n: f"{L}/ready_{n}"
SIM = [(f"gen_{n}", f"{PY}/paper_gen.py {n}", []) for n in R2_ORDER] + \
      [(f"gen_{n}", f"{PY}/paper_gen.py {n}", []) for n in R3_CELLS + ["MIS_E2o", "MIS_E2o_nooff"]]
HEAVY = [(f"r1_{n}", f"{PY}/paper_r1.py {n}", []) for n in STAGE2] + \
        [(f"r2b_{n}", f"{PY}/paper_r2.py ladder {n}", [rd(n)]) for n in R2_ORDER] + \
        [(f"r3_{n}", f"{PY}/paper_r3.py {n}", [rd(n)]) for n in R3_CELLS] + \
        [("r4_synth", f"{PY}/paper_r4.py synth", []), ("r4_real", f"{PY}/paper_r4.py real", [])] + \
        [("r5_synth", f"{PY}/paper_r5.py synth", []), ("r5_real", f"{PY}/paper_r5.py real", [rd("MIS_E2o"), rd("MIS_E2o_nooff")])] + \
        [(f"r6_{n}", f"{PY}/paper_r6.py {n}", []) for n in ["E1_nu50_gain_iid100", "E2_cup_iid100", "E3_fetch_iid100"]] + \
        [("r2c", f"{PY}/paper_r2.py validate E1o_P", [rd("E1o_P")])]
kind = sys.argv[1]
tasks = SIM if kind == "sim" else HEAVY
while True:
    todo = [t for t in tasks if not os.path.exists(f"{CL}/{t[0]}.done") and not os.path.isdir(f"{CL}/{t[0]}")]
    if not todo:
        print(kind, "queue empty", flush=True)
        break
    ready = [t for t in todo if all(os.path.exists(p) for p in t[2])]
    if not ready:
        time.sleep(30)
        continue
    name, cmd, _ = ready[0]
    try:
        os.mkdir(f"{CL}/{name}")
    except FileExistsError:
        continue
    t0 = time.time()
    with open(f"{L}/{name}.log", "a") as f:
        rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if rc == 0:
        open(f"{CL}/{name}.done", "w").write(f"{time.time() - t0:.0f}")
    else:
        os.rmdir(f"{CL}/{name}")           # crashed: release (will be retried only after a fix and a restart)
        open(f"{L}/{name}.failed", "w").write(str(rc))
        time.sleep(5)
        tasks = [t for t in tasks if t[0] != name]
