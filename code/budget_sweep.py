"""Budget cliff: continuous-mode energy of A' vs the best uniform design (B1) as the LUT
budget changes (model estimates; accuracy threshold from acc_model.json).

For each budget, every accurate configuration gets its energy-optimal design point (mode,
target cycles, per-node variants) at the fastest clock; A' is the minimum over all accurate
configurations and B1 over the uniform ones.

  python budget_sweep.py --tables tables.json --acc acc_model.json --out budget_sweep.json
"""

import argparse
import json

import numpy as np

import energy_search as es
from assemble_costs import DEVICE, RES, assemble, choose_variants, load_tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--acc", required=True)
    p.add_argument("--lut", type=float, nargs="+",
                   default=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.add_argument("--bram-offset", type=float, default=0.1, help="BRAM derate = LUT - this")
    p.add_argument("--power", nargs="+", default=["nominal", "z2_anchor"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    table, grid = load_tables(args.tables)
    a = json.load(open(args.acc))
    cfgs = [b for b in es.ALL_CONFIGS if a["acc"][",".join(map(str, b))] >= a["threshold"]]
    fmax = max(es.F_SET)
    out = {"threshold": a["threshold"], "n_configs": len(cfgs), "f_mhz": fmax, "budgets": []}
    for lut in args.lut:
        budget = {"LUT": DEVICE["LUT"] * lut,
                  "BRAM_18K": DEVICE["BRAM_18K"] * min(1.0, lut - args.bram_offset),
                  "DSP": DEVICE["DSP"]}
        pts = []   # (config, cycles, U)
        for b in cfgs:
            for mode in ("lut", "dsp", "lut_finn"):
                for T in grid:
                    nodes = assemble(table, list(b), mode, T)
                    if nodes is None:
                        continue
                    if mode == "lut_finn":
                        tot = {k: sum(n["auto"][k] for n in nodes) for k in RES}
                        if any(tot[k] > budget[k] for k in RES):
                            continue
                    else:
                        _, tot = choose_variants(nodes, budget, es.UTIL_W)
                        if tot is None:
                            continue
                    pts.append((b, max(n["cycles"] for n in nodes),
                                es.utilisation(tot["LUT"], tot["BRAM_18K"], tot["DSP"])))
        row = {"lut": lut, "bram": round(budget["BRAM_18K"] / DEVICE["BRAM_18K"], 2)}
        for pname in args.power:
            pp = es.POWER[pname]
            e = np.array([(pp["P0"] / fmax + pp["k_f"] + (pp["k_fU"] + pp["k_act"]) * U) * cyc * 1e-6
                          for _, cyc, U in pts])
            uni = np.array([es.is_uniform(b) for b, _, _ in pts])
            ia, ib = int(np.argmin(e)), int(np.argmin(np.where(uni, e, np.inf)))
            i4 = int(np.argmin([cyc for _, cyc, _ in pts]))
            row[pname] = {"A_prime_mJ": round(float(e[ia]) * 1e3, 4),
                          "A_prime": "".join(map(str, pts[ia][0])), "A_prime_fps": round(fmax * 1e6 / pts[ia][1]),
                          "B1_mJ": round(float(e[ib]) * 1e3, 4), "B1": "".join(map(str, pts[ib][0])),
                          "B1_fps": round(fmax * 1e6 / pts[ib][1]),
                          "B4_mJ": round(float(e[i4]) * 1e3, 4),
                          "B1_over_A_pct": round(100 * (e[ib] / e[ia] - 1), 1),
                          "B4_over_A_pct": round(100 * (e[i4] / e[ia] - 1), 1)}
        out["budgets"].append(row)
        r = row[args.power[0]]
        print(f"LUT {lut:.2f}: A' {r['A_prime']} {r['A_prime_fps']} FPS {r['A_prime_mJ']} mJ | "
              f"B1 {r['B1']} {r['B1_fps']} FPS +{r['B1_over_A_pct']}% | B4 +{r['B4_over_A_pct']}%",
              flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
