"""Vivado-calibrated LUT cost of FINN v0.10.1's RTL matrix-vector unit (replaces the lo/hi bracket).

FINN estimates 0 LUT for MVAU_rtl. The cost tables bracket it: lo = only the LUTRAM of
distributed weights, hi = FINN's estimate of the same layer as a DSP-based HLS MVAU
(cost_tables.rtl_mvu_luts). Vivado out-of-context synthesis of the 53 audited units
(audit_rtl_ops.py vivado) gives the real count, so:

  prepare  rebuild the audited units (same configurations and foldings as audit_rtl_ops.py)
           and store each unit's hi estimate (block weights) next to its Vivado LUTs
  fit      least-squares LUT_vivado = k * hi + b, with leave-one-out error, against the
           bracket ends
  tables   write a cost table whose MVAU_rtl LUTs are k * hi + b (+ the LUTRAM of distributed
           weights), from a table whose LUT field holds hi (tables_thr_hi.json)

  python calib_mvu_vivado.py prepare --audit audit
  python calib_mvu_vivado.py fit --audit audit
  python calib_mvu_vivado.py tables --audit audit --tables cost/tables_thr_hi.json --out cost/tables_viv.json
"""

import argparse
import json
import os
import tempfile

import numpy as np


def cmd_prepare(args):
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    import audit_rtl_ops as a
    import cost_tables as ct
    path = os.path.join(args.audit, "audit.json")
    cases = json.load(open(path))
    by_sig = {tuple(sorted(c["attrs"].items())): c for c in cases if c["op"] == "MVAU_rtl"}
    for bits in a.CONFIGS:
        m0 = ct.hw_model(bits, True, tempfile.mkdtemp())
        for T in a.T_LIST:
            m = m0.transform(SetFolding(T, mvau_wwidth_max=1024, two_pass_relaxation=False))
            m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
            m = m.transform(InferDataTypes())
            for n in m.graph.node:
                if n.op_type != "MVAU_rtl":
                    continue
                sig = tuple(sorted(dict(a.signature(getCustomOp(n))).items()))
                c = by_sig.get(sig)
                if c is not None and "bracket" not in c:
                    luts = ct.rtl_mvu_luts(n)
                    c["bracket"] = {"lo": luts["block"][0], "hi": luts["block"][1],
                                    "lutram": luts["distributed"][0]}
    json.dump(cases, open(path, "w"), indent=1)
    done = sum(1 for c in by_sig.values() if "bracket" in c)
    print(f"bracket stored for {done} of {len(by_sig)} MVAU_rtl cases")


def load_points(audit):
    cs = [c for c in json.load(open(os.path.join(audit, "audit.json")))
          if c["op"] == "MVAU_rtl" and "bracket" in c and c.get("vivado", {}).get("returncode") == 0]
    hi = np.array([c["bracket"]["hi"] for c in cs])
    viv = np.array([c["vivado"]["LUT"] for c in cs])
    return cs, hi, viv


def fit(hi, viv):
    X = np.stack([hi, np.ones_like(hi)], axis=1)
    k, b = np.linalg.lstsq(X, viv, rcond=None)[0]
    loo = []
    for i in range(len(hi)):
        mask = np.arange(len(hi)) != i
        ki, bi = np.linalg.lstsq(X[mask], viv[mask], rcond=None)[0]
        loo.append(abs(ki * hi[i] + bi - viv[i]) / max(viv[i], 1.0))
    return float(k), float(b), float(np.median(loo)), float(np.max(loo))


def cmd_fit(args):
    cs, hi, viv = load_points(args.audit)
    k, b, med, worst = fit(hi, viv)
    print(f"{len(cs)} units: LUT_vivado = {k:.3f} * hi + {b:.0f}; leave-one-out error median "
          f"{100 * med:.1f}%, worst {100 * worst:.1f}%")
    print(f"Vivado / hi: median {np.median(viv / hi):.2f} (range {np.min(viv / hi):.2f}-{np.max(viv / hi):.2f}); "
          f"total Vivado {viv.sum():.0f} vs hi {hi.sum():.0f} vs lo 0 (block weights)")
    yos = np.array([c["yosys"]["LUT"] for c in cs])
    print(f"Yosys / Vivado: median {np.median(yos / viv):.2f}")


def cmd_tables(args):
    _, hi, viv = load_points(args.audit)
    k, b, _, _ = fit(hi, viv)
    d = json.load(open(args.tables))
    n_set = 0
    for rec in d["table"].values():
        for n in rec["nodes"]:
            if n["op"] != "MVAU_rtl" or "LUT_hi" not in n["auto"]:
                continue
            for r in [n["auto"]] + n.get("variants", []):
                # LUT holds hi; for distributed weights hi includes their LUTRAM
                lutram = r["LUT_hi"] - n["auto"]["LUT_hi"] if r is not n["auto"] else 0.0
                r["LUT"] = k * (r["LUT_hi"] - lutram) + b + lutram
                n_set += 1
    d["mvu_lut_calibration"] = {"k": k, "b": b}
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n_set} MVAU_rtl entries set to {k:.3f} * hi + {b:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "fit", "tables"])
    p.add_argument("--audit", required=True)
    p.add_argument("--tables")
    p.add_argument("--out")
    args = p.parse_args()
    {"prepare": cmd_prepare, "fit": cmd_fit, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
