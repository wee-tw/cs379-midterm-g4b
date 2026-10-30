"""Audit FINN v0.10.1's resource estimates of its RTL operators against a Yosys synthesis.

  generate  fold the ResNet-8 hardware model (standalone thresholds, so MVUs become RTL) at a
            few bit configurations and cycle targets, and for every distinct RTL node
            (Thresholding_rtl, ConvolutionInputGenerator_rtl, FMPadding_rtl, MVAU_rtl) write
            FINN's own HDL (generate_hdl) and its estimate (node_res_estimation) to a case dir
  synth     synthesize every case with Yosys (sv-elab frontend, synth_xilinx -family xc7) and
            compare: LUT = logic LUTs + LUTRAM and SRL cells counted as LUTs, BRAM18 = RAMB18 +
            2 x RAMB36, DSP; Xilinx primitives instantiated by the RTL are blackboxes (--prims)

  vivado    the same cases through Vivado out-of-context synthesis (xc7z020), the reference
            for both: report_utilization's Slice LUTs (logic + memory), Block RAM tiles x 2, DSPs

Yosys maps logic and memories differently from Vivado, so its LUT/BRAM numbers are indicative;
counts of instantiated primitives (DSP48E1) are exact. Set PYTHONUTF8=1 on Windows: FINN
reads its RTL templates with the default encoding.

  python audit_rtl_ops.py generate --out audit
  python audit_rtl_ops.py synth --out audit --yosys .../yosys.exe --prims xilinx_prims_bb.sv
  python audit_rtl_ops.py vivado --out audit --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile

PART = "xc7z020clg400-1"
OPS = ("Thresholding_rtl", "ConvolutionInputGenerator_rtl", "FMPadding_rtl", "MVAU_rtl")
CONFIGS = ([4, 4, 4, 4, 4, 4, 4, 8], [8, 4, 2, 4, 2, 4, 4, 8], [2, 2, 2, 2, 2, 2, 2, 8],
           [8, 8, 8, 8, 8, 8, 8, 8])
T_LIST = (100000, 25000, 12500)
# LUT equivalents of the distributed-RAM and shift-register cells synth_xilinx may emit
LUTRAM = {"RAM32M": 4, "RAM64M": 4, "RAM32M16": 8, "RAM64M8": 8, "RAM32X1D": 2, "RAM64X1D": 2,
          "RAM128X1D": 4, "RAM256X1S": 4, "RAM32X1S": 1, "RAM64X1S": 1, "RAM128X1S": 2,
          "SRL16E": 1, "SRLC32E": 1}
LOGIC = ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "LUT6_2")
FF = ("FDRE", "FDSE", "FDCE", "FDPE")


def signature(inst):
    keep = ("PE", "SIMD", "MW", "MH", "NumChannels", "ram_style", "inputDataType",
            "outputDataType", "weightDataType", "IFMDim", "OFMDim", "ConvKernelDim", "Stride",
            "Padding", "ImgDim")
    attrs = inst.get_nodeattr_types()
    return tuple((k, str(inst.get_nodeattr(k))) for k in keep if k in attrs)


def cmd_generate(args):
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    import cost_tables as ct
    seen, cases = set(), []
    for bits in CONFIGS:
        m0 = ct.hw_model(bits, True, tempfile.mkdtemp())
        for T in T_LIST:
            m = m0.transform(SetFolding(T, mvau_wwidth_max=1024, two_pass_relaxation=False))
            m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
            m = m.transform(InferDataTypes())
            for n in m.graph.node:
                if n.op_type not in OPS:
                    continue
                inst = getCustomOp(n)
                sig = (n.op_type,) + signature(inst)
                if sig in seen:
                    continue
                seen.add(sig)
                name = f"{n.op_type}_{len(cases):03d}"
                d = os.path.join(args.out, name)
                os.makedirs(d, exist_ok=True)
                inst.set_nodeattr("code_gen_dir_ipgen", os.path.abspath(d))
                inst.generate_hdl(m, PART, 10.0)
                est = {k: float(v) for k, v in inst.node_res_estimation(PART).items()
                       if k in ("LUT", "BRAM_18K", "URAM", "DSP")}
                cases.append({"case": name, "op": n.op_type, "bits": "".join(map(str, bits)),
                              "T": T, "attrs": dict(sig[1:]), "finn_estimate": est})
                print(name, dict(sig[1:]).get("PE"), dict(sig[1:]).get("SIMD"), est, flush=True)
    json.dump(cases, open(os.path.join(args.out, "cases.json"), "w"), indent=1)
    print(len(cases), "cases")


def top_module(case_dir, op):
    pat = {"Thresholding_rtl": "*_axi_wrapper.v", "ConvolutionInputGenerator_rtl": "*_wrapper.v",
           "FMPadding_rtl": "FMPadding_rtl_*.v", "MVAU_rtl": "*_wrapper.v"}[op]
    files = [f for f in glob.glob(os.path.join(case_dir, pat)) if not f.endswith("_sim.v")]
    m = re.search(r"^\s*module\s+(\w+)", open(files[0], encoding="utf-8").read(), re.M)
    return m.group(1)


def case_sources(case_dir, op, mvu_rtl):
    """HDL of one case: FINN's generated files, or for MVAU_rtl the mvu RTL plus the wrapper."""
    if op == "MVAU_rtl":
        srcs = sorted(glob.glob(os.path.join(mvu_rtl, "*.sv"))) + \
               glob.glob(os.path.join(case_dir, "*_wrapper.v"))
    else:
        # packages first: Vivado reads the files in order and the SWG implementation imports swg_pkg
        srcs = sorted(glob.glob(os.path.join(case_dir, "*.sv")), key=lambda s: (not s.endswith("_pkg.sv"), s))
        srcs += sorted(f for f in glob.glob(os.path.join(case_dir, "*.v")) if not f.endswith("_sim.v"))
    return [s.replace("\\", "/") for s in srcs]


def cmd_synth(args):
    cases = json.load(open(os.path.join(args.out, "cases.json")))
    root = os.path.dirname(os.path.dirname(os.path.abspath(args.yosys)))
    os.environ["PATH"] = os.pathsep.join([os.path.join(root, "bin"), os.path.join(root, "lib"),
                                          os.environ.get("PATH", "")])
    for c in cases:
        d = os.path.abspath(os.path.join(args.out, c["case"]))
        srcs = [args.prims.replace("\\", "/")] + case_sources(d, c["op"], args.mvu_rtl)
        top = top_module(d, c["op"])
        with open(os.path.join(d, "synth.ys"), "w") as f:
            # the SWG wrapper uses parameters before declaring them, which Vivado accepts
            f.write(f"read_slang --allow-use-before-declare --top {top} " + " ".join(srcs) + "\n")
            f.write(f"synth_xilinx -family xc7 -top {top} -flatten -nodsp\nstat\n")
        r = subprocess.run([args.yosys, "-m", "slang", "-q", "-l", "synth.log", "-s", "synth.ys"],
                           cwd=d, capture_output=True, text=True)
        text = open(os.path.join(d, "synth.log"), encoding="utf-8", errors="replace").read()
        block = text[text.rfind("Printing statistics"):] if "Printing statistics" in text else ""
        cnt = {k: int(v) for v, k in re.findall(r"^\s+(\d+)\s+\$?(\w+)\s*$", block, re.M)}
        luts = sum(cnt.get(k, 0) for k in LOGIC)
        lutram = sum(n * cnt.get(k, 0) for k, n in LUTRAM.items())
        c["yosys"] = {"returncode": r.returncode, "LUT_logic": luts, "LUT_ram_srl": lutram,
                      "LUT": luts + lutram, "FF": sum(cnt.get(k, 0) for k in FF),
                      "BRAM18": cnt.get("RAMB18E1", 0) + 2 * cnt.get("RAMB36E1", 0),
                      "DSP": cnt.get("DSP48E1", 0)}
        e, y = c["finn_estimate"], c["yosys"]
        print(f"{c['case']:34s} rc={r.returncode} PE/SIMD {c['attrs'].get('PE')}/{c['attrs'].get('SIMD')} "
              f"LUT finn {e.get('LUT', 0):7.0f} yosys {y['LUT']:6d} | BRAM18 finn {e.get('BRAM_18K', 0):4.0f} "
              f"yosys {y['BRAM18']:3d} | DSP finn {e.get('DSP', 0):4.0f} yosys {y['DSP']:3d}", flush=True)
    json.dump(cases, open(os.path.join(args.out, "audit.json"), "w"), indent=1)
    # summary by operator: median ratio yosys / finn and share of zero estimates
    print("\n| operator | cases | FINN LUT = 0 | median LUT yosys/finn | BRAM18 finn vs yosys (sum) | DSP finn vs yosys (sum) |")
    print("|---|---|---|---|---|---|")
    for op in OPS:
        cs = [c for c in cases if c["op"] == op and c.get("yosys", {}).get("returncode") == 0]
        if not cs:
            continue
        zero = sum(1 for c in cs if c["finn_estimate"].get("LUT", 0) == 0)
        ratios = [c["yosys"]["LUT"] / c["finn_estimate"]["LUT"] for c in cs if c["finn_estimate"].get("LUT", 0) > 0]
        med = f"{sorted(ratios)[len(ratios) // 2]:.1f}x" if ratios else "n/a"
        bf = sum(c["finn_estimate"].get("BRAM_18K", 0) for c in cs)
        by = sum(c["yosys"]["BRAM18"] for c in cs)
        df = sum(c["finn_estimate"].get("DSP", 0) for c in cs)
        dy = sum(c["yosys"]["DSP"] for c in cs)
        print(f"| {op} | {len(cs)} | {zero} | {med} | {bf:.0f} vs {by} | {df:.0f} vs {dy} |")


def vivado_case(c, args):
    """Out-of-context Vivado synthesis of one case; returns its utilisation. Runs in a copy
    under a short work directory (Vivado fails on Windows paths over 260 characters)."""
    src = os.path.abspath(os.path.join(args.out, c["case"]))
    d = os.path.join(args.workdir, c["case"])
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".Xil", "*.log", "*.ys"))
    top = top_module(d, c["op"])
    with open(os.path.join(d, "vivado_synth.tcl"), "w") as f:
        for s in case_sources(d, c["op"], args.mvu_rtl):
            f.write(f"read_verilog {'-sv ' if s.endswith('.sv') else ''}{{{s}}}\n")
        f.write(f"synth_design -top {top} -part {PART} -mode out_of_context\n")
        f.write("report_utilization -file vivado_util.rpt\n")
    rpt = os.path.join(d, "vivado_util.rpt")
    if os.path.exists(rpt):
        os.remove(rpt)
    # cwd = case dir: the threshold RTL reads its $readmemh files by relative path
    r = subprocess.run([args.vivado, "-mode", "batch", "-nojournal", "-log", "vivado.log",
                        "-source", "vivado_synth.tcl"], cwd=d, capture_output=True, text=True)
    text = open(rpt).read() if os.path.exists(rpt) else ""
    for f in ("vivado_util.rpt", "vivado.log"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), src)

    def cell(name):
        m = re.search(r"^\|\s*" + name + r"\*?\s*\|\s*([\d.]+)", text, re.M)
        return float(m.group(1)) if m else 0.0
    return {"returncode": r.returncode if text else -1, "LUT": cell("Slice LUTs"),
            "LUT_logic": cell("LUT as Logic"), "LUT_mem": cell("LUT as Memory"),
            "FF": cell("Slice Registers"), "BRAM18": 2 * cell("Block RAM Tile"), "DSP": cell("DSPs")}


def cmd_vivado(args):
    """Vivado (xc7z020, out of context) as the reference for the FINN and Yosys numbers."""
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(args.out, "audit.json")
    cases = json.load(open(path))
    with ThreadPoolExecutor(args.jobs) as ex:
        for c, v in zip(cases, ex.map(lambda c: vivado_case(c, args), cases)):
            c["vivado"] = v
            e, y = c["finn_estimate"], c.get("yosys", {})
            print(f"{c['case']:34s} rc={v['returncode']} LUT finn {e.get('LUT', 0):7.0f} yosys "
                  f"{y.get('LUT', 0):6d} vivado {v['LUT']:6.0f} | BRAM18 {e.get('BRAM_18K', 0):3.0f} / "
                  f"{y.get('BRAM18', 0):3d} / {v['BRAM18']:4.1f} | DSP {e.get('DSP', 0):4.0f} / "
                  f"{y.get('DSP', 0):3d} / {v['DSP']:3.0f}", flush=True)
    json.dump(cases, open(path, "w"), indent=1)
    print("\n| operator | cases | LUT finn / yosys / vivado | BRAM18 finn / yosys / vivado | DSP finn / yosys / vivado |")
    print("|---|---|---|---|---|")
    for op in OPS:
        cs = [c for c in cases if c["op"] == op and c["vivado"]["returncode"] == 0]
        if cs:
            tot = lambda src, k: sum(c[src].get(k, 0) for c in cs)  # noqa: E731
            print(f"| {op} | {len(cs)} | {tot('finn_estimate', 'LUT'):.0f} / {tot('yosys', 'LUT'):.0f} / "
                  f"{tot('vivado', 'LUT'):.0f} | {tot('finn_estimate', 'BRAM_18K'):.0f} / "
                  f"{tot('yosys', 'BRAM18'):.0f} / {tot('vivado', 'BRAM18'):.0f} | "
                  f"{tot('finn_estimate', 'DSP'):.0f} / {tot('yosys', 'DSP'):.0f} / {tot('vivado', 'DSP'):.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "synth", "vivado"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--yosys")
    ap.add_argument("--prims")
    ap.add_argument("--mvu-rtl", help="finn-rtllib/mvu of the FINN version that generated the cases")
    ap.add_argument("--vivado", help="vivado executable (vivado.bat on Windows)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--workdir", default="D:/va", help="short path to run Vivado in")
    args = ap.parse_args()
    {"generate": cmd_generate, "synth": cmd_synth, "vivado": cmd_vivado}[args.cmd](args)


if __name__ == "__main__":
    main()
