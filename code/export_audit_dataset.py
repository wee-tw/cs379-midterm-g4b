"""Export the estimator audits as CSV files with a README (../results/audit_dataset/; notes 12.32).

  rtl_nodes.csv     159 FINN v0.10.1 RTL nodes (audit_rtl_ops.py): FINN estimate, Yosys 0.70
                    and Vivado 2022.2 out-of-context synthesis (xc7z020clg400-1)
  hls_nodes.csv     MVAU_hls with internal thresholds in three ram_style_thresholds styles
                    (audit_hls_thresholds.py): FINN estimate, Vitis HLS csynth estimate, and the
                    export_design -flow syn synthesis (random-init weights; trained weights where run)
  thr_triggers.csv  Thresholding_rtl with depth_trigger_bram = 2^30 and 1024 (thr_trigger_vivado.py):
                    the threshold-memory cost model against Vivado
  designs.csv       every node of the selected designs (verify_designs.py; "set" names the output
                    folder): v1-calibrated prediction against synthesis, with vectorless dynamic
                    power at the design clock

  python export_audit_dataset.py --scratch <scratchpad> --designs D:/vd D:/vd2 --out ../results/audit_dataset
"""

import argparse
import csv
import glob
import json
import os
import re


def write(path, rows):
    if not rows:
        return 0
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def csynth(case_dir):
    f = glob.glob(os.path.join(case_dir, "MVAU_hls_*_csynth.rpt"))
    if not f:
        return {}
    m = re.search(r"\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", open(f[0], encoding="utf-8", errors="replace").read())
    return {"hls_csynth_BRAM18": int(m.group(1)), "hls_csynth_DSP": int(m.group(2)),
            "hls_csynth_FF": int(m.group(3)), "hls_csynth_LUT": int(m.group(4))} if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--designs", nargs="+", default=["D:/vd"], help="verify_designs.py output folders")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = args.scratch
    rtl = []
    for c in json.load(open(f"{S}/audit/audit.json")):
        r = {"case": c["case"], "op": c["op"], "bits_config": c.get("bits"), "T": c.get("T"), **c.get("attrs", {})}
        r.update({f"finn_{k}": v for k, v in c["finn_estimate"].items()})
        r.update({f"yosys_{k}": v for k, v in (c.get("yosys") or {}).items() if k != "returncode"})
        r.update({f"vivado_{k}": v for k, v in (c.get("vivado") or {}).items() if k != "returncode"})
        rtl.append(r)
    hls = []
    for audit, weights in (("audit_hls_finn", "random"), ("audit_hls", "random"), ("audit_hls_trained", "trained")):
        p = f"{S}/{audit}/audit_hls.json"
        if not os.path.exists(p):
            continue
        for c in json.load(open(p)):
            s = c.get("synth") or {}
            r = {"audit": audit, "weights": weights, "case": c["case"], "bits_config": c.get("bits"),
                 "ram_style_thresholds": c["style"], **c.get("attrs", {})}
            r.update({f"finn_{k}": v for k, v in c["finn_estimate"].items()})
            r.update(csynth(os.path.join(S, audit, c["case"])))
            r.update({f"synth_{k}": v for k, v in s.items() if k != "returncode"})
            hls.append(r)
    thr = []
    p = f"{S}/audit_thr_trig/thr_trigger.json"
    if os.path.exists(p):
        for j in json.load(open(p)):
            thr.append({"case": j["case"], "depth_trigger_bram": j["D"], **j["params"],
                        "model_LUT": j["model"][0], "model_BRAM18": j["model"][1],
                        **{f"vivado_{k}": v for k, v in j["vivado"].items() if k != "returncode"}})
    des = []
    for dd in args.designs:
        if not os.path.exists(f"{dd}/results.json"):
            continue
        res = json.load(open(f"{dd}/results.json"))
        for c in json.load(open(f"{dd}/cases.json")):
            v = res.get(c["case"], {})
            des.append({"set": os.path.basename(dd.rstrip("/")), "design": c["design"], "role": c["role"],
                        "case": c["case"], "op": c["op"],
                        "kind": c["kind"], "f_MHz": c["f_MHz"], **{f"pred_{k}": x for k, x in c["predicted"].items()},
                        **{f"synth_{k}": x for k, x in v.items() if k not in ("returncode", "same_as")},
                        "synthesized": v.get("returncode") == 0})
    n = {name: write(os.path.join(args.out, name), rows) for name, rows in
         (("rtl_nodes.csv", rtl), ("hls_nodes.csv", hls), ("thr_triggers.csv", thr), ("designs.csv", des))}
    print(n)


if __name__ == "__main__":
    main()
