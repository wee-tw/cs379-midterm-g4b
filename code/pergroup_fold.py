"""Per-group folding targets: can a design keep its throughput (II) but cut its single-frame
latency L by folding the non-bottleneck groups faster? (model estimates)

FINN's SetFolding gives every layer the same cycle target T, so L is roughly the sum of
39 nodes each close to T. The cost tables store every group at every T, so any mix
(T_G1, T_G2, T_G3, T_G4) can be assembled exactly like the uniform-T design points:
II = max node cycles, L = critical path (latency_check.crit_from_nodes, which matches
FINN's dataflow_performance), resources = per-node variant choice inside the budget.

  python pergroup_fold.py --bits 4 4 4 4 4 4 8 8 --mode lut --out cost/pergroup_44444488.json
"""

import argparse
import itertools
import json
import time

import numpy as np

import energy_search as es
from assemble_costs import DEVICE, RES, choose_variants, load_tables
from cost_tables import contexts
from latency_check import crit_from_nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs=8, default=[4, 4, 4, 4, 4, 4, 8, 8])
    p.add_argument("--mode", default="lut")
    p.add_argument("--tables", default="cost/v0/tables.json")
    p.add_argument("--lat-scale", type=float, default=0.6024)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    table, grid = load_tables(args.tables)
    budget = {k: DEVICE[k] * es.DERATE[k] for k in RES}
    groups = []
    for g, (w, a, pr) in enumerate(contexts(args.bits)):
        groups.append({T: table[f"{args.mode}|{T}|{g}|{w}|{a}|{pr}"]["nodes"] for T in grid})
    rows, t0 = [], time.time()
    for Ts in itertools.product(grid, repeat=4):
        nodes = [n for g, T in enumerate(Ts) for n in groups[g][T]]
        pick, tot = choose_variants(nodes, budget, es.UTIL_W)
        if pick is None:
            continue
        rows.append({"T": list(Ts), "II": max(n["cycles"] for n in nodes),
                     "L": crit_from_nodes(nodes)[0],
                     "U": round(float(es.utilisation(tot["LUT"], tot["BRAM_18K"], tot["DSP"])), 4),
                     "LUT": tot["LUT"], "BRAM": tot["BRAM_18K"], "DSP": tot["DSP"]})
    print(f"{len(rows)} feasible mixes in {time.time() - t0:.0f} s")
    uni = {r["II"]: r for r in rows if len(set(r["T"])) == 1}
    out = {"bits": args.bits, "mode": args.mode, "levels": []}
    for II in sorted({r["II"] for r in rows}):
        cand = [r for r in rows if r["II"] <= II]
        best = min(cand, key=lambda r: (r["L"], r["U"]))
        u = min((r for r in rows if len(set(r["T"])) == 1 and r["II"] <= II),
                key=lambda r: (r["L"], r["U"]), default=None)
        if u is None:
            continue
        lev = {"II": II, "uniform_T": u, "best_mix": best,
               "L_cut_pct": round(100 * (1 - best["L"] / u["L"]), 1)}
        out["levels"].append(lev)
        print(f"II <= {II:7d}: uniform T {u['T'][0]:7d} L {u['L']:8d} U {u['U']:.3f} | best mix "
              f"T {best['T']} II {best['II']} L {best['L']:8d} U {best['U']:.3f} "
              f"(L -{lev['L_cut_pct']}%)")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
