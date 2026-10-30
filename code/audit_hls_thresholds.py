"""Do FINN's MVAU_hls estimates cover the thresholds? Synthesize the same HLS node three times.

FINN v0.10.1 counts an MVAU_hls's threshold LUTRAM and its 2^B - 1 comparators only for
ram_style_thresholds = "distributed". For "auto" (the default) and "block" (HLS pragma
ROM_2P_BRAM) both terms are 0, and bram_estimation never counts thresholds
(implementation_notes.md 12.27). This script measures what the three styles really cost.

  generate  lut-mode ResNet-8 models (thresholds inside the MVAUs) at a few bit widths and one
            folding; for every distinct MVAU_hls with thresholds and each threshold style,
            write FINN's HLS code (code_generation_ipgen) and FINN's estimate. resType is set
            to "lut" so that FINN's estimate and the HLS code agree on the multipliers.
  run       Vitis HLS: csynth, then export_design -flow syn (Vivado synthesis of the HLS RTL
            for xc7z020), and parse the post-synthesis utilisation

  python audit_hls_thresholds.py generate --out audit_hls
  python audit_hls_thresholds.py run --out audit_hls --finn-root <finn-v0.10.1> \
      --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat
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
CONFIGS = ([4, 8, 4, 8, 4, 8, 4, 8], [4, 4, 4, 4, 4, 4, 4, 8], [2, 2, 2, 2, 2, 2, 2, 8])
T = 25000
STYLES = ("auto", "block", "distributed")


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
    configs = [list(map(int, c)) for c in args.configs] if args.configs else CONFIGS
    for bits, t in [(b, t) for b in configs for t in (args.T or [T])]:
        ckpt = dict(c.split("=", 1) for c in args.ckpt or []).get("".join(map(str, bits)))
        m = ct.hw_model(bits, False, tempfile.mkdtemp(), ckpt=ckpt)
        m = m.transform(SetFolding(t, mvau_wwidth_max=args.wwidth, two_pass_relaxation=False))
        m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
        m = m.transform(InferDataTypes())
        for n in m.graph.node:
            if n.op_type != "MVAU_hls":
                continue
            inst = getCustomOp(n)
            if inst.get_nodeattr("noActivation") == 1:
                continue
            attrs = {k: str(inst.get_nodeattr(k)) for k in
                     ("PE", "SIMD", "MW", "MH", "inputDataType", "weightDataType",
                      "outputDataType", "accDataType", "mem_mode")}
            sig = tuple(attrs.items())
            if sig in seen:
                continue
            seen.add(sig)
            inst.set_nodeattr("resType", "lut")
            idx = len(cases) // len(STYLES)
            for style in STYLES:
                inst.set_nodeattr("ram_style_thresholds", style)
                name = f"{n.name}_{idx:02d}_{style}"
                d = os.path.join(args.out, name)
                os.makedirs(d, exist_ok=True)
                inst.set_nodeattr("code_gen_dir_ipgen", os.path.abspath(d))
                inst.code_generation_ipgen(m, PART, 10.0)
                est = {"LUT": float(inst.lut_estimation()), "BRAM_18K": float(inst.bram_estimation()),
                       "DSP": float(inst.dsp_estimation(PART))}
                cases.append({"case": name, "node": n.name, "bits": "".join(map(str, bits)),
                              "style": style, "attrs": attrs, "finn_estimate": est})
                print(name, attrs["PE"], attrs["SIMD"], attrs["outputDataType"], attrs["mem_mode"],
                      est, flush=True)
    json.dump(cases, open(os.path.join(args.out, "cases.json"), "w"), indent=1)
    print(len(cases), "cases")


def parse_util(case_dir):
    """Post-synthesis utilisation from Vivado's report inside the HLS export (Slice LUTs,
    Block RAM tiles x 2, DSPs); None if the export did not run."""
    rpts = glob.glob(os.path.join(case_dir, "project_*", "sol1", "impl", "**",
                                  "*utilization_synth.rpt"), recursive=True)
    if not rpts:
        return None
    text = open(rpts[0]).read()

    def cell(name):
        m = re.search(r"^\|\s*" + name + r"\*?\s*\|\s*([\d.]+)", text, re.M)
        return float(m.group(1)) if m else 0.0
    return {"LUT": cell("Slice LUTs"), "LUT_logic": cell("LUT as Logic"),
            "LUT_mem": cell("LUT as Memory"), "FF": cell("Slice Registers"),
            "BRAM18": 2 * cell("Block RAM Tile"), "DSP": cell("DSPs")}


def run_case(c, args):
    # Vivado (inside export_design) fails on paths over 260 characters, so each case runs in a
    # copy under a short work directory; its reports are copied back next to the case
    src = os.path.abspath(os.path.join(args.out, c["case"]))
    d = os.path.join(args.workdir, c["case"])
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("project_*", "*.log"))
    tcl = [f for f in glob.glob(os.path.join(d, "hls_syn_*.tcl")) if not f.endswith("_audit.tcl")][0]
    text = open(tcl).read().replace("export_design -format ip_catalog",
                                    "export_design -flow syn -rtl verilog -format ip_catalog")
    # FINN writes the Windows code-gen path with backslashes, which Tcl reads as escapes
    text = re.sub(r'(set config_hwsrcdir ")([^"]*)"',
                  lambda m: m.group(1) + d.replace("\\", "/") + '"', text)
    open(os.path.join(d, "hls_syn_audit.tcl"), "w").write(text)
    env = dict(os.environ, FINN_ROOT=args.finn_root)
    r = subprocess.run([args.vitis_hls, "-f", "hls_syn_audit.tcl"], cwd=d, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    open(os.path.join(d, "vitis_hls_run.log"), "w", encoding="utf-8").write(
        (r.stdout or "") + (r.stderr or ""))
    for f in glob.glob(os.path.join(d, "project_*", "sol1", "**", "*.rpt"), recursive=True):
        if "utilization" in f or f.endswith("csynth.rpt") or "export" in os.path.basename(f):
            shutil.copy(f, src)
    for f in ("vitis_hls.log", "vitis_hls_run.log"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), src)
    return {"returncode": r.returncode, **(parse_util(d) or {"LUT": None})}


def cmd_run(args):
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(args.out, "cases.json")
    cases = json.load(open(path))
    if args.only:
        cases = [c for c in cases if re.search(args.only, c["case"])]
    with ThreadPoolExecutor(args.jobs) as ex:
        for c, v in zip(cases, ex.map(lambda c: run_case(c, args), cases)):
            c["synth"] = v
            e = c["finn_estimate"]
            print(f"{c['case']:32s} rc={v['returncode']} LUT finn {e['LUT']:7.0f} synth {v.get('LUT')} | "
                  f"BRAM18 finn {e['BRAM_18K']:4.0f} synth {v.get('BRAM18')} | DSP {e['DSP']:3.0f} / {v.get('DSP')}",
                  flush=True)
    name = "audit_hls.json" if not args.only else "audit_hls_subset.json"
    json.dump(cases, open(os.path.join(args.out, name), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--vitis-hls", help="vitis_hls executable (vitis_hls.bat on Windows)")
    ap.add_argument("--finn-root", help="FINN source tree with deps/finn-hlslib")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", help="regex on case names (run a subset)")
    ap.add_argument("--configs", nargs="*", help="bit strings, e.g. 44444448 (default CONFIGS)")
    ap.add_argument("--T", type=int, nargs="*", help="SetFolding targets (default 25000)")
    ap.add_argument("--ckpt", nargs="*", help="bits=checkpoint.pt: trained weights and thresholds")
    ap.add_argument("--wwidth", type=int, default=1024,
                    help="SetFolding mvau_wwidth_max (36 = FINN's default, as in lut_finn)")
    ap.add_argument("--workdir", default="D:/ha", help="short path to run in (Windows path limit)")
    args = ap.parse_args()
    {"generate": cmd_generate, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
