import sys
sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import energy_search as es
from assemble_costs import DEVICE, RES, assemble, load_tables, options, util
mode_fix = sys.argv[2]
table, grid = load_tables(sys.argv[1])
budget = {"LUT": DEVICE["LUT"] * 0.7, "BRAM_18K": DEVICE["BRAM_18K"] * 0.6, "DSP": DEVICE["DSP"] * 1.0}
nodes = assemble(table, [8, 8, 4, 2, 2, 4, 2, 2], "lut", 25000)
opts = [options(n) for n in nodes]
if mode_fix == "dedup":
    new = []
    for os_ in opts:
        seen, keep = set(), []
        for o in os_:
            key = tuple(o[k] for k in RES)
            if key not in seen:
                seen.add(key); keep.append(o)
        new.append(keep)
    opts = new
w = es.UTIL_W
idx = [(i, j) for i, os_ in enumerate(opts) for j in range(len(os_))]
c = np.array([util(opts[i][j], w) for i, j in idx])
A_one = np.zeros((len(nodes), len(idx)))
for col, (i, j) in enumerate(idx):
    A_one[i, col] = 1
A_res = np.array([[opts[i][j][k] for (i, j) in idx] for k in RES])
cons = [LinearConstraint(A_one, 1, 1), LinearConstraint(A_res, -np.inf, [budget[k] for k in RES])]
kw = {"options": {"presolve": False}} if mode_fix == "nopresolve" else {}
res = milp(c, constraints=cons, integrality=np.ones(len(idx)), bounds=Bounds(0, 1), **kw)
print(mode_fix, res.success, res.status, round(res.fun, 6) if res.success else None)
