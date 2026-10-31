"""Baseline gaps with Vivado's dynamic power in place of the model's (notes 12.45).

Per design: E = (P0 + k_f*f + P_dyn) * cycles / (f * 1e6), where P_dyn is the sum of the per-node
`report_power` dynamic power of the design at its clock (verify_designs.py report lines), and P0, k_f
and the cycles are the model's. Designs are identified by bits; their mode, T and clock come from the
design lists of the synthesis runs.

  python vivado_power_gaps.py --table main_table_tsr_v1218_m0.json \
      --reports vd6_report.txt vd7_report.txt [vd8_report.txt] --extra 44444448_lut_finn_16667=0.854@100 \
      --designs vd6_designs.json vd7_designs.json [vd8_designs.json] \
      --points 125=points_f125.npz 111.11=points_f111.11.npz 100=points_f100.npz
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_search as es  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--designs", nargs="+", required=True, help="design lists [bits, mode, T, f, role, ...]")
    ap.add_argument("--extra", nargs="*", default=[], help="<bits>_<mode>_<T>=<W>@<MHz> for designs synthesized elsewhere")
    ap.add_argument("--points", nargs="+", required=True)
    args = ap.parse_args()
    pdyn = {}
    for rep in args.reports:
        for line in open(rep, encoding="utf-8", errors="replace"):
            m = re.match(r"== (\S+) \(.*dynamic power ([\d.]+) W at ([\d.]+) MHz", line)
            if m:
                pdyn[m.group(1)] = float(m.group(2))
    for e in args.extra:
        tag, rest = e.split("=")
        pdyn[tag] = float(rest.split("@")[0])
    design = {}
    for f in args.designs:
        for d in json.load(open(f)):
            design.setdefault(d[0], (d[1], int(d[2]), float(d[3])))
    if "44444448" not in design:
        design["44444448"] = ("lut_finn", 16667, 100.0)   # B2 at margin 1 (verify_b2lut)
    points = {float(k): np.load(v)["rows"] for k, v in (s.split("=", 1) for s in args.points)}
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    table = json.load(open(args.table))

    def energy(bits, p):
        mode, T, f = design[bits]
        tag = f"{bits}_{mode}_{T}"
        if tag not in pdyn:
            return None, tag
        rows = points[f]
        mask = (rows[:, 0].astype(int) == idx[bits]) & (rows[:, 1] == es.MODES.index(mode)) & (rows[:, 2] == T)
        cyc = rows[mask][0, 3]
        pp = es.POWER[p]
        return (pp["P0"] + pp["k_f"] * f + pdyn[tag]) * cyc / (f * 1e6) * 1e3, tag

    for p in ("nominal", "z2_anchor"):
        for m in sorted(table, key=float):
            r = table[m][p]
            ea, ta = energy(r["A_verified"]["bits"], p)
            out = [f"A' {ta} {'-' if ea is None else f'{ea:.4f} mJ'}"]
            for b in ("B1_measured", "B2_measured"):
                bits = (r.get(b) or {}).get("bits")
                if not bits or bits not in design:
                    out.append(f"{b[:2]} {bits}: no design")
                    continue
                eb, tb = energy(bits, p)
                model = 100 * r[b]["rel_to_A"]
                viv = "-" if (eb is None or ea is None) else f"{100 * (eb / ea - 1):+.1f}%"
                out.append(f"{b[:2]} {bits}: model {model:+.1f}% -> Vivado {viv}")
            print(f"{p:9s} margin {float(m):g}: " + " | ".join(out))


if __name__ == "__main__":
    main()
