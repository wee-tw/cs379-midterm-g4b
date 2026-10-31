import sys
sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
import numpy as np
import energy_search as es
from assemble_costs import DEVICE, RES, assemble, load_tables, options, util
table, grid = load_tables(sys.argv[1])
budget = {"LUT": DEVICE["LUT"] * 0.7, "BRAM_18K": DEVICE["BRAM_18K"] * 0.6, "DSP": DEVICE["DSP"] * 1.0}
nodes = assemble(table, [8, 8, 4, 2, 2, 4, 2, 2], "lut", 25000)
opts = [options(n) for n in nodes]
print("nodes", len(nodes), "options per node", [len(o) for o in opts])
for k in RES:
    lo = sum(min(o[k] for o in os_) for os_ in opts)
    print(k, "min possible", round(lo, 1), "budget", budget[k])
sep = [int(np.argmin([util(o, es.UTIL_W) for o in os_])) for os_ in opts]
print("separable totals", {k: round(sum(opts[i][sep[i]][k] for i in range(len(nodes))), 1) for k in RES})
# any node with a single option or identical options?
for i, os_ in enumerate(opts):
    vals = [tuple(round(o[k], 3) for k in RES) for o in os_]
    if len(set(vals)) < len(vals):
        print("node", i, nodes[i].get("name", ""), "has duplicate options", vals)
