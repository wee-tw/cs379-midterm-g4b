"""Resource weights of the power model from Vivado's vectorless estimates (12.32; tool estimates).

The energy model's utilisation U = LUT/LUT_dev + w_BRAM * BRAM/BRAM_dev + w_DSP * DSP/DSP_dev uses
assumed weights (0.25, 0.25; uw_sens.py checked 0.1-1.0). verify_designs.py synthesized every node
of five selected designs and ran report_power (vectorless, 12.5% toggle) at the design clock.
Here the dynamic power of the distinct synthesized units is fitted as a nonnegative combination
of LUT, BRAM18 and DSP counts (normalized to 125 MHz), the weights are turned into U's units,
and the main search is rerun with them (energies recomputed on the calibrated points).
Vivado's power estimate is itself about 20% off a board measurement (PowerGear, DATE 2022), so
this checks the ordering of LUT, BRAM and DSP cost, not absolute power.

  python power_weights.py --designs D:/vd D:/vd2 --points cost/points_calv2.npz --acc cost/acc_model_3rep.json       --out cost/power_weights_calv2.json
"""

import argparse
import json

import numpy as np
from scipy.optimize import nnls

import energy_search as es
from assemble_costs import DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True, help="verify_designs.py output folders")
    ap.add_argument("--points", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pts = []
    for dd in args.designs:
        cases = json.load(open(f"{dd}/cases.json"))
        res = json.load(open(f"{dd}/results.json"))
        pts += [(res[c["case"]], c) for c in cases
                if res.get(c["case"], {}).get("returncode") == 0 and res[c["case"]].get("power_W") is not None
                and "same_as" not in res[c["case"]]]
    X = np.array([[v["LUT"], v["BRAM18"], v["DSP"]] for v, _ in pts], float)
    y = np.array([v["power_W"] * 125.0 / c["f_MHz"] for v, c in pts])
    coef, _ = nnls(X, y)
    r2 = 1 - np.sum((X @ coef - y) ** 2) / np.sum((y - y.mean()) ** 2)
    a, c_bram, d_dsp = coef
    w_bram = c_bram * DEVICE["BRAM_18K"] / (a * DEVICE["LUT"])
    w_dsp = d_dsp * DEVICE["DSP"] / (a * DEVICE["LUT"])
    print(f"{len(pts)} synthesized units: dynamic power at 125 MHz = {1e6 * a:.1f} uW/LUT + "
          f"{1e3 * c_bram:.2f} mW/BRAM18 + {1e3 * d_dsp:.2f} mW/DSP (R^2 {r2:.2f})")
    print(f"in U's units (a full device of each, relative to LUT): w_BRAM = {w_bram:.2f}, w_DSP = {w_dsp:.2f}"
          f" (model assumes 0.25, 0.25)")
    rows = np.load(args.points)["rows"]
    m = json.load(open(args.acc))
    acc = [m["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS]
    out = {"fit": {"uW_per_LUT": 1e6 * a, "mW_per_BRAM18": 1e3 * c_bram, "mW_per_DSP": 1e3 * d_dsp,
                   "r2": r2, "n": len(pts), "w_BRAM": w_bram, "w_DSP": w_dsp}, "search": {}}
    for label, w in (("assumed", (0.25, 0.25)), ("vivado", (w_bram, w_dsp))):
        es.UTIL_W.update({"BRAM_18K": w[0], "DSP": w[1]})
        for mg in (1, 2, 3, 5, 8):
            s = es.search(rows, acc, m["reference_acc"] - mg, es.POWER["nominal"])["continuous"]
            e0 = s["A_prime"]["E_continuous_mJ"]
            line = {k: (''.join(map(str, d["bits"])), d["mode"], round(100 * (d["E_continuous_mJ"] / e0 - 1), 1))
                    for k, d in s.items()}
            out["search"][f"{label}|{mg}"] = line
            print(f"{label:8s} margin {mg}: A' {line['A_prime'][0]} {line['A_prime'][1]} "
                  f"{e0:.4f} mJ | B1 {line['B1_uniform'][2]:+.1f}% B2 {line.get('B2_proxy', ('', '', None))[2]}% "
                  f"B3 {line.get('B3_sequential', ('', '', None))[2]}% B4 {line['B4_latency'][2]:+.1f}%")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
