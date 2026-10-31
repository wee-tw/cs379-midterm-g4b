"""Width-converter dataset: FINN's DWC LUT estimate against synthesis, one row per width pair (notes 12.46).

Reads the dwc_final.py cases and verify_designs.py results copied into results/timing and writes
results/audit_dataset/dwc_synth.csv. `uses` counts the converters of all listed sets with that width pair.

  python dwc_dataset.py --sets "final designs=vdw" "A' with MW576 narrowed=vdw_narrow" \
      "A' with MW576 narrowed=vdw_narrow_m5"
"""

import argparse
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="<source label>=<prefix>, reading results/timing/<prefix>_cases.json and _results.json")
    ap.add_argument("--out", default=os.path.join(R, "audit_dataset", "dwc_synth.csv"))
    args = ap.parse_args()
    rows = {}
    for s in args.sets:
        tag, prefix = s.split("=", 1)
        cases = json.load(open(os.path.join(R, "timing", f"{prefix}_cases.json")))
        res = json.load(open(os.path.join(R, "timing", f"{prefix}_results.json")))
        for c in cases:
            r = res.get(c["case"])
            if not r or r.get("returncode") != 0:
                continue
            k = (c["op"], int(c["attrs"]["inWidth"]), int(c["attrs"]["outWidth"]))
            e = rows.setdefault(k, {"impl": c["op"].split("_")[-1], "in_bits": k[1], "out_bits": k[2],
                                    "direction": "up" if k[1] < k[2] else "down",
                                    "integer_ratio": max(k[1:]) % min(k[1:]) == 0,
                                    "finn_lut_estimate": round(c["predicted"]["LUT"]), "synth_lut": r["LUT"],
                                    "synth_ff": r.get("FF"), "period_ns": r.get("period_ns"), "uses": 0,
                                    "source": tag})
            e["uses"] += 1
    out = sorted(rows.values(), key=lambda e: (e["impl"], e["direction"], e["in_bits"], e["out_bits"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "distinct width pairs written to", args.out)


if __name__ == "__main__":
    main()
