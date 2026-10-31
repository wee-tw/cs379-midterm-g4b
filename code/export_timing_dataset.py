"""Timing part of the audit dataset (implementation_notes.md 12.35-12.36): two CSV files next
to the resource audit in results/audit_dataset/.

  timing_hls.csv  every synthesized MVAU_hls (hls_timing_scan.py): shape, folding, widths,
                  resType, the HLS clock target and the post-synthesis period (Vitis HLS 2022.2
                  export_design -flow syn = target - WNS of Vivado's synthesis)
  timing_rtl.csv  the RTL units timed by rtl_timing.py: shape, folding, widths and the
                  post-synthesis period with the clock known to synthesis (--constrain-synth)
                  and without it (clock added after synthesis)

  python export_timing_dataset.py --hls-roots D:/vd ... --rtl-constrained c.json --rtl-plain p.json \
      --out ../results/audit_dataset
"""

import argparse
import csv
import json
import os

from hls_timing_scan import scan


def bits(dtype):
    return int("".join(ch for ch in dtype if ch.isdigit())) if dtype else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hls-roots", nargs="+", required=True)
    ap.add_argument("--rtl-constrained", required=True)
    ap.add_argument("--rtl-plain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hls = scan(args.hls_roots)
    with open(os.path.join(args.out, "timing_hls.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "case", "MW", "MH", "PE", "SIMD", "SF", "weight_bits", "input_bits",
                    "thresholds_in_node", "resType", "target_ns", "period_ns"])
        for r in hls:
            w.writerow([r["root"], r["case"], r["MW"], r["MH"], r["PE"], r["SIMD"], r["MW"] // r["SIMD"],
                        r["W"], r["A"], int(r["TMEM"] > 0), r["res"], r["target"], r["period"]])
    plain = {x["case"]: x for x in json.load(open(args.rtl_plain))}
    rtl = json.load(open(args.rtl_constrained))
    with open(os.path.join(args.out, "timing_rtl.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "op", "MW", "MH", "PE", "SIMD", "weight_bits", "input_bits", "clock_ns",
                    "period_ns_clock_in_synthesis", "period_ns_clock_after_synthesis"])
        for x in rtl:
            a = x["attrs"]
            w.writerow([x["case"], x["op"], a.get("MW"), a.get("MH"), a.get("PE"), a.get("SIMD"),
                        bits(a.get("weightDataType")), bits(a.get("inputDataType")), x["period_ns"],
                        x["achieved_ns"], plain.get(x["case"], {}).get("achieved_ns")])
    print(f"timing_hls.csv {len(hls)} rows, timing_rtl.csv {len(rtl)} rows")


if __name__ == "__main__":
    main()
