"""Print the continuous-regime selections (A', B1-B4) with resources, per margin, for a points file."""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/115-1/embedded sys midterm report/code")
from energy_search import ALL_CONFIGS, POWER, search  # noqa: E402

S = "C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
rows = np.load(sys.argv[1])["rows"]
m = json.load(open(f"{S}/cost/acc_model_3rep.json"))
acc = [m["acc"][",".join(map(str, b))] for b in ALL_CONFIGS]
for mg in (1, 2, 3, 5, 8):
    thr = m["reference_acc"] - mg
    res = search(rows, acc, thr, POWER[sys.argv[2] if len(sys.argv) > 2 else "nominal"])["continuous"]
    e0 = res["A_prime"]["E_continuous_mJ"]
    print(f"== margin {mg} (thr {thr:.2f})")
    for k, d in res.items():
        print(f"  {k:14s} {''.join(map(str, d['bits']))} {d['mode']:8s} T {d['T_target']:7d} f {d['f_MHz']:6.2f} "
              f"FPS {d['FPS']:8.1f} LUT {d['LUT']:6d} BRAM18 {d['BRAM18']:5.1f} DSP {d['DSP']:3d} U {d['U']:.3f} "
              f"E {d['E_continuous_mJ']:.4f} mJ ({100 * (d['E_continuous_mJ'] / e0 - 1):+.1f}%) acc {d['acc_est']}")
