"""How much do the main results move with the error of the LUT calibrations? (12.31; estimates)

The calibrated tables (tables_calov) fit the RTL-MVU LUTs to Vivado and the HLS-MVU LUTs to
Vitis HLS with a leave-one-out error of about 11% (median). This scales one of the two by a
factor (e.g. 0.85 or 1.15, a systematic error larger than the median per-node error) and the
main pipeline is rerun on the result (points, threshold_sweep):

  tables   write a copy of a cost table with every MVAU_rtl / MVAU_hls LUT entry scaled
  summary  per scenario and margin: A' bits and implementation, A' energy against the
           unscaled run, and the B1 / B2 / B3 / B4 gaps

  python calib_sensitivity.py tables --tables cost/tables_calov.json --rtl 1.15 --out cost/t.json
  python calib_sensitivity.py summary --base cost/threshold_sweep_calov.json \
      --runs rtl0.85=cost/threshold_sweep_s1.json rtl1.15=...
"""

import argparse
import json


def cmd_tables(args):
    d = json.load(open(args.tables))
    scale = {"MVAU_rtl": args.rtl, "MVAU_hls": args.hls}
    n = 0
    for rec in d["table"].values():
        for node in rec["nodes"]:
            s = scale.get(node["op"], 1.0) * args.all
            if s == 1.0:
                continue
            for r in [node["auto"]] + node.get("variants", []):
                r["LUT"] = s * r["LUT"]
                n += 1
    d["lut_scale"] = {**scale, "all": args.all}
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n} entries scaled, MVAU_rtl x{args.rtl}, MVAU_hls x{args.hls}, all x{args.all}")


def cmd_summary(args):
    base = json.load(open(args.base))
    runs = [("calibrated", base)] + [(k, json.load(open(v))) for k, v in (r.split("=", 1) for r in args.runs)]
    gaps = ("B1_uniform", "B2_proxy", "B3_sequential", "B4_latency")
    print("| scenario | margin | A' bits | B1 / B2 / B3 / B4 above A' |")
    print("|---|---|---|---|")
    for name, d in runs:
        for mg, r in d.items():
            c = r[args.power]["cont"]
            bits = r[args.power]["cont_bits"]
            print(f"| {name} | {float(mg):g} | {bits['A_prime']} | "
                  + " / ".join(f"{100 * c[g]:+.1f}%" if c.get(g) is not None else "-" for g in gaps) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["tables", "summary"])
    p.add_argument("--tables")
    p.add_argument("--rtl", type=float, default=1.0)
    p.add_argument("--hls", type=float, default=1.0)
    p.add_argument("--all", type=float, default=1.0, help="scale on every node's LUTs (design-level error)")
    p.add_argument("--out")
    p.add_argument("--base")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--power", default="nominal")
    args = p.parse_args()
    {"tables": cmd_tables, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
