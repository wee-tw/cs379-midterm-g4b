import sys
import numpy as np
from energy_search import ALL_CONFIGS, F_SET, MODES, POWER, energies, utilisation
rows = np.load(sys.argv[1])["rows"]
cfg = rows[:, 0].astype(int)
idx = {"".join(map(str, b)): i for i, b in enumerate(ALL_CONFIGS)}
fps, e_c, _, _ = energies(rows, POWER["nominal"], max(F_SET))
for b in sys.argv[2:]:
    m = np.where(cfg == idx[b])[0]
    order = m[np.argsort(e_c[m])]
    print(f"== {b}: {len(m)} design points; best 4 by continuous energy (nominal, 125 MHz)")
    # distinct cycle levels and the cheapest point at each
    for i in order[:4]:
        ci, mi, T, cyc, lut, bram, dsp = rows[i]
        print(f"   E {e_c[i]*1e3:.4f} mJ  mode {MODES[int(mi)]:8s} T {int(T):6d} II {int(cyc):6d} "
              f"FPS {fps[i]:7.0f} LUT {int(lut):6d} BRAM18 {bram:5.1f} DSP {int(dsp):3d} "
              f"U {utilisation(lut, bram, dsp):.3f}")
    levels = sorted(set(rows[m, 3].astype(int)))
    print("   reachable II levels:", levels[:12])
