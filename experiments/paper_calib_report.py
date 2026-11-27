"""Summarise the new keys of results/cwk/CALIBRATION.json into results/paper/CALIBRATION.md (and print the table)."""
import sys, json
sys.path.insert(0, "/home/user/world_models")
cal = json.load(open("/home/user/world_models/results/cwk/CALIBRATION.json"))
OLD = {"e1_nu50_gain", "e1_nu80_gain", "e1_nu50_sat", "e1_nu50_deadzone", "e2", "e3"}
rows, force, mis = [], [], []
for k, v in cal.items():
    if k in OLD:
        continue
    if k.startswith("info_"):
        ver = v["verify"]
        rows.append((k, v["env"], v["law"], v["pi"] if v["law"] == "gated" else "-", v["r"], v["iid"], v["sigma"], round(v["gain"], 4),
                     round(v["dev_oracle"] * 100, 1), round(ver["oracle"] * 100, 1), round(ver["copy"] * 100, 1), round(ver["rho"], 2),
                     "yes" if v["in_window"] else "NO", ("yes" if v["capped"] else "no") + (" (refined)" if v.get("refined") else "")))
    elif k.startswith("force_unit"):
        force.append((k, round(v["sigma_ref"], 4), round(v["force"], 4), round(v["unit"], 4)))
    elif k.startswith("misalign"):
        ver = v["verify"]
        mis.append((k, v["sigma"], round(v["gain"], 4), round(v["dev_oracle"] * 100, 1), round(ver["oracle"] * 100, 1), round(ver["copy"] * 100, 1),
                    round(ver["rho"], 2), "yes" if v["in_window"] else "NO"))
md = ["# Calibration of the paper cells (simulator-only development cohorts)\n",
      "Target (authors' design): outcome-blind oracle R^2 about 50% (+- 8) on the primary group. Search: per-axis command budget sigma "
      "(command gain g = 0.98 (1 - sigma) / action limit), secant on 40 parents x 24 draws development cohorts (seed offset 500000); "
      "verification on a second disjoint cohort of 48 parents x 32 draws (offset 600000). MISALIGN: offsets 300000 / 400000. No model, "
      "no method comparison enters. New keys in results/cwk/CALIBRATION.json; the Stage 2 keys are untouched.\n",
      "## Information-limited cells (INFOLIM)\n", "| key | env | law | pi | r | iid | sigma | gain | dev oracle % | verification oracle % | verification copy % | rho | within 50 +- 8 | at search bound |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
md += ["\n## Force unit (r = d + 2 cells: force scale = unit x sigma; force-only driver calibrated to oracle R^2 0.70)\n", "| key | sigma_ref | force scale | unit |", "|---|---|---|---|"]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in force]
md += ["\n## MISALIGN (Ball-in-Cup, rotate channel [+ offset], tanh law)\n", "| key | sigma | gain | dev oracle % | verification oracle % | verification copy % | rho | within 50 +- 8 |", "|---|---|---|---|---|---|---|---|"]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in mis]
open("/home/user/world_models/results/paper/CALIBRATION.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
