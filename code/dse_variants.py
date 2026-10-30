"""Four cost models for the same DSP-mode designs, to attribute DSE harm to each estimator gap.

All four start from the dspc records of the corrected cost tables (cost_tables.py, mode
dspc: FINN's SetFolding at each T, standalone thresholds, MVAU_rtl for >= 4-bit layers), so
the designs, foldings and implementation variants are identical; only the MVAU_rtl costs
differ:
  finn       DSP = ceil(PE/4) * SIMD (FINN's dsp_estimation), LUT = 0 (FINN's lut_estimation)
  dsp_fixed  DSP as the RTL instantiates it, LUT = 0
  lut_fixed  DSP as FINN estimates, LUT = the high bound (DSP-based HLS MVAU estimate)
  reference  DSP as instantiated, LUT = the high bound (close to a Yosys synthesis, 12.21)
lut and lut_finn records (no RTL MVUs) are copied unchanged. Writes tables_<variant>.json
under the mode name dspc, so energy_search.py points / dse_harm.py treat them alike.

  python dse_variants.py --base cost/tables.json --corr cost/tables_c2.json --outdir cost
"""

import argparse
import copy
import json
import math
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="tables.json (lut, lut_finn)")
    p.add_argument("--corr", required=True, help="tables_c2.json (dspc with LUT_hi)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    base = json.load(open(args.base))
    corr = json.load(open(args.corr))
    keep = {k: v for k, v in base["table"].items() if k.split("|")[0] in ("lut", "lut_finn")}
    dspc = {k: v for k, v in corr["table"].items() if k.startswith("dspc|")}
    for variant in ("finn", "dsp_fixed", "lut_fixed", "reference"):
        table = dict(keep)
        for k, rec in dspc.items():
            rec = copy.deepcopy(rec)
            for n in rec["nodes"]:
                if n["op"] != "MVAU_rtl":
                    continue
                pe, simd = n["fold"]["PE"], n["fold"]["SIMD"]
                finn_dsp = float(math.ceil(pe / 4) * simd)
                for r in [n["auto"]] + n.get("variants", []):
                    true_dsp, hi = r["DSP"], r["LUT_hi"]
                    r["DSP"] = true_dsp if variant in ("dsp_fixed", "reference") else finn_dsp
                    r["LUT"] = hi if variant in ("lut_fixed", "reference") else 0.0
            table[k] = rec
        out = os.path.join(args.outdir, f"tables_{variant}.json")
        json.dump({"T_GRID": base["T_GRID"], "modes": ["lut", "lut_finn", "dspc"],
                   "table": table, "mismatches": []}, open(out, "w"))
        print("wrote", out, len(table), "records")


if __name__ == "__main__":
    main()
