"""Independent check of timing_safe.py sweep: per point, its clock class is the highest class
whose point set contains it; its continuous energy is the minimum over every clock <= class.
A', B1 and B2 are then recomputed directly and compared with the merged sweep JSON."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:/115-1/embedded sys midterm report/code")
import energy_search as es  # noqa: E402

C = r"C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/cost"
TAG = sys.argv[1] if len(sys.argv) > 1 else "tsr_v2"
CLASSES = (125.0, 111.11, 100.0, 90.91, 83.33, 76.92, 71.43, 66.67, 50.0)
CLOCKS = sorted(set(CLASSES) | set(es.F_SET_FULL))

cls = {}
for c in sorted(CLASSES):                       # ascending: higher classes overwrite
    for r in np.load(f"{C}/points_{TAG}_f{c:g}.npz")["rows"]:
        cls[(int(r[0]), int(r[1]), int(r[2]))] = c
rows = np.load(f"{C}/points_{TAG}_f50.npz")["rows"]
pc = np.array([cls[(int(r[0]), int(r[1]), int(r[2]))] for r in rows])
a = json.load(open(f"{C}/acc_model_3rep.json"))
acc = np.array([a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS])
uni = np.array([es.is_uniform(b) for b in es.ALL_CONFIGS])
cfg = rows[:, 0].astype(int)
merged = json.load(open(f"{C}/threshold_sweep_{TAG}_merged.json"))
worst = 0.0
for pname in ("nominal", "z2_anchor"):
    p = es.POWER[pname]
    e = np.full(len(rows), np.inf)
    for f in CLOCKS:
        ef = es.energies(rows, p, f)[1]
        e = np.where(f <= pc + 1e-9, np.minimum(e, ef), e)
    fps100 = es.energies(rows, p, 100.0)[0]
    for m in (1.0, 2.0, 3.0, 5.0, 8.0):
        thr = a["reference_acc"] - m
        ok = acc[cfg] >= thr
        ia = int(np.argmin(np.where(ok, e, np.inf)))
        ib1 = int(np.argmin(np.where(ok & uni[cfg], e, np.inf)))
        accs = [c for c in range(len(es.ALL_CONFIGS)) if acc[c] >= thr]
        b2 = min(accs, key=lambda c: (es.bops(es.ALL_CONFIGS[c]), -acc[c]))
        mb2 = (cfg == b2) & (rows[:, 1] == es.MODES.index("lut_finn"))
        ib2 = int(np.argmax(np.where(mb2, fps100, -np.inf)))
        e_b2 = es.energies(rows[ib2:ib2 + 1], p, 100.0)[1][0]
        mine = {"A": e[ia] * 1e3, "B1": e[ib1] / e[ia] - 1, "B2": e_b2 / e[ia] - 1}
        mc = merged[str(m)][pname]["continuous"]
        theirs = {"A": mc["A_prime"]["E_mJ"], "B1": mc["B1_uniform"]["rel_to_A"], "B2": mc["B2_proxy"]["rel_to_A"]}
        bits_ok = "".join(map(str, es.ALL_CONFIGS[cfg[ia]])) == mc["A_prime"]["bits"]
        d = max(abs(mine["A"] - theirs["A"]) / theirs["A"], abs(mine["B1"] - theirs["B1"]), abs(mine["B2"] - theirs["B2"]))
        worst = max(worst, d)
        print(f"{pname:9s} margin {m:g}: A' {mine['A']:.4f} vs {theirs['A']:.4f} mJ, bits same {bits_ok} | "
              f"B1 {100 * mine['B1']:+.1f}% vs {100 * theirs['B1']:+.1f}% | B2 {100 * mine['B2']:+.1f}% vs {100 * theirs['B2']:+.1f}%")
print("largest difference:", round(worst, 6))
