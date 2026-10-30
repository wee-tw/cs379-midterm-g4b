"""Synthesize the designs FINN's estimates pick, to show they do not fit (12.25, 12.27).

The DSE-harm study found that with FINN's estimates the energy-optimal design at accuracy
margins 2-8 is a dspc design that needs more DSP slices than the xc7z020 has. This rebuilds
those designs exactly (standalone thresholds, SetFolding at their T, FINN's own minimisation
passes), writes FINN's HDL for every RTL matrix-vector unit, and synthesizes each one out of
context with Vivado. DSP48E1s are instantiated by the RTL, so their sum over the units is the
design's DSP use (the other nodes use none).

  python prove_infeasible.py generate --out infeasible
  python prove_infeasible.py vivado --out infeasible --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
      --mvu-rtl <mvu_v0.10.1>
"""

import argparse
import json
import os
import tempfile

PART = "xc7z020clg400-1"
DEVICE_DSP = 220
# FINN-estimate optima of the threshold-consistent harm study (dse_harm_thr.json, nominal)
DESIGNS = (("84242448", 12500, "margin 2"), ("84222448", 14286, "margin 3"),
           ("84222244", 14286, "margins 5 and 8"))


def cmd_generate(args):
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    import cost_tables as ct
    cases = []
    for bits, T, role in DESIGNS:
        m = ct.hw_model(list(map(int, bits)), True, tempfile.mkdtemp())
        m = m.transform(SetFolding(T, mvau_wwidth_max=1024, two_pass_relaxation=False))
        m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
        m = m.transform(InferDataTypes())
        finn_total = 0.0
        for n in m.graph.node:
            if not n.domain.startswith("finn.custom_op.fpgadataflow"):
                continue
            inst = getCustomOp(n)
            est = inst.node_res_estimation(PART)
            finn_total += float(est.get("DSP", 0))
            if n.op_type != "MVAU_rtl":
                continue
            name = os.path.join(bits, n.name)
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            inst.set_nodeattr("code_gen_dir_ipgen", os.path.abspath(d))
            inst.generate_hdl(m, PART, 10.0)
            cases.append({"case": name, "op": "MVAU_rtl", "design": bits, "T": T, "role": role,
                          "PE": inst.get_nodeattr("PE"), "SIMD": inst.get_nodeattr("SIMD"),
                          "W": inst.get_weight_datatype().bitwidth(),
                          "A": inst.get_input_datatype().bitwidth(),
                          "finn_dsp": float(est.get("DSP", 0)),
                          "rtl_dsp": float(ct.rtl_mvu_dsps(inst))})
        print(bits, T, role, "FINN DSP total", finn_total,
              "MVAU_rtl:", [(c["case"], c["PE"], c["SIMD"], c["finn_dsp"], c["rtl_dsp"])
                            for c in cases if c["design"] == bits], flush=True)
        cases.append({"case": None, "design": bits, "finn_dsp_total": finn_total})
    json.dump(cases, open(os.path.join(args.out, "cases.json"), "w"), indent=1)


def cmd_vivado(args):
    from concurrent.futures import ThreadPoolExecutor
    import audit_rtl_ops as a
    cases = json.load(open(os.path.join(args.out, "cases.json")))
    units = [c for c in cases if c.get("case")]
    with ThreadPoolExecutor(args.jobs) as ex:
        for c, v in zip(units, ex.map(lambda c: a.vivado_case(c, args), units)):
            c["vivado"] = v
            print(c["case"], "PE", c["PE"], "SIMD", c["SIMD"], f"W{c['W']}A{c['A']}",
                  "DSP finn", c["finn_dsp"], "rtl", c["rtl_dsp"], "vivado", v["DSP"], "LUT", v["LUT"],
                  flush=True)
    json.dump(cases, open(os.path.join(args.out, "infeasible.json"), "w"), indent=1)
    print("\n| design | role | FINN DSP (whole design) | Vivado DSP (RTL MVUs) | device |")
    print("|---|---|---|---|---|")
    for bits, T, role in DESIGNS:
        tot = next(c["finn_dsp_total"] for c in cases if c.get("case") is None and c["design"] == bits)
        viv = sum(c["vivado"]["DSP"] for c in units if c["design"] == bits)
        print(f"| {bits} (T {T}) | {role} | {tot:.0f} | {viv:.0f} | {DEVICE_DSP} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "vivado"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vivado")
    ap.add_argument("--mvu-rtl")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--workdir", default="D:/vi")
    args = ap.parse_args()
    {"generate": cmd_generate, "vivado": cmd_vivado}[args.cmd](args)


if __name__ == "__main__":
    main()
