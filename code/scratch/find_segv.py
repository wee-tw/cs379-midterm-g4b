import sys, faulthandler
faulthandler.enable()
sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
import energy_search as es
from assemble_costs import DEVICE, RES, assemble, choose_variants, load_tables
table, grid = load_tables(sys.argv[1])
budget = {"LUT": DEVICE["LUT"] * 0.7, "BRAM_18K": DEVICE["BRAM_18K"] * 0.6, "DSP": DEVICE["DSP"] * 1.0}
lo, hi = int(sys.argv[2]), int(sys.argv[3])
for ci in range(lo, hi):
    b = es.ALL_CONFIGS[ci]
    for mode in ("lut", "dsp"):
        for T in grid:
            nodes = assemble(table, list(b), mode, T)
            if nodes is None:
                continue
            print(ci, b, mode, T, flush=True)
            choose_variants(nodes, budget, es.UTIL_W)
print("no crash in", lo, hi)
