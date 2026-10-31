import sys
import numpy as np
from energy_search import ALL_CONFIGS, F_SET, POWER, energies
rows = np.load(sys.argv[1])["rows"]
cfg = rows[:, 0].astype(int)
for pname in ("nominal", "z2_anchor"):
    _, e_c, _, _ = energies(rows, POWER[pname], max(F_SET))
    best = np.full(len(ALL_CONFIGS), np.inf)
    np.minimum.at(best, cfg, e_c)
    idx = {"".join(map(str, b)): i for i, b in enumerate(ALL_CONFIGS)}
    print(pname, " ".join(f"{b}={best[idx[b]] * 1e3:.4f}" for b in sys.argv[2:]))
