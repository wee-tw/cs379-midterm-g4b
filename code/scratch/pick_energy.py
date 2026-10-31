"""Continuous-regime energy-optimal design of given configurations (bits strings) in a points file."""
import json, sys
import numpy as np
sys.path.insert(0, "D:/115-1/embedded sys midterm report/code")
from energy_search import ALL_CONFIGS, POWER, search  # noqa: E402
rows = np.load(sys.argv[1])["rows"]
for bits in sys.argv[2:]:
    acc = [100.0 if "".join(map(str, b)) == bits else 0.0 for b in ALL_CONFIGS]
    d = search(rows, acc, 50.0, POWER["nominal"])["continuous"]["A_prime"]
    print(f"{bits} {d['mode']:8s} T {d['T_target']:6d} f {d['f_MHz']:.0f} FPS {d['FPS']:8.1f} LUT {d['LUT']:6d} "
          f"BRAM18 {d['BRAM18']:5.1f} DSP {d['DSP']:3d} E {d['E_continuous_mJ']:.4f} mJ")
