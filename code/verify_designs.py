"""Synthesize every node of the selected designs (implementation_notes.md 12.32; tool-verified).

The calibrated cost model (tables_calov: Vivado/HLS fits, replay-buffer BRAM, RTL DSP rule)
picks the designs of the main results. Here each picked design is rebuilt exactly as the
cost tables build it and every node is synthesized out of context, so the model's whole-design
prediction can be checked on designs it was not fitted on:

  generate  hw_model(bits), SetFolding at T (plus the DSP-aware refold for dsp4c), minimize
            widths; apply the per-node choices of the search (assemble_costs.choose_variants on
            the table; lut_finn keeps FINN's defaults); write FINN's code per node: RTL HDL
            (generate_hdl), HLS (code_generation_ipgen), and for every decoupled MVU its weight
            streamer (finn-rtllib memstream with the node's memblock.dat). A threshold node
            whose pick is the all-LUTRAM variant gets depth_trigger_bram = 2^30, which makes
            the RTL force distributed RAM as the model assumes (trigger 0 would leave it to
            Vivado)
  run       RTL nodes and weight streamers: Vivado synth_design -mode out_of_context, then
            report_utilization and a vectorless report_power at the design's clock; HLS nodes:
            Vitis HLS csynth + export_design -flow syn, then report_power on its checkpoint
  report    predicted vs synthesized LUT / BRAM18 / DSP per design and per operator, and the
            summed dynamic power (FIFOs and width converters are in neither: the budget
            derating covers them)

  python verify_designs.py generate --tables cost/tables_calov.json --out D:/vd
  python verify_designs.py run --out D:/vd --vivado .../vivado.bat --vitis-hls .../vitis_hls.bat \
      --finn-root <finn-v0.10.1> --mvu-rtl <mvu_v0.10.1> --jobs 4
  python verify_designs.py report --out D:/vd
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
# (bits, mode, T, f_MHz, role): the calibrated picks at margin 1-2 and the margin-1 baselines
DESIGNS = (("44444448", "lut", 14286, 125.0, "A' margin 1"),
           ("44444448", "dsp4c", 16667, 125.0, "A' margin 1, 5% LUT headroom"),
           ("44242448", "dsp4c", 14286, 125.0, "A' margin 2"),
           ("84848484", "lut", 16667, 125.0, "B1 margin 1"),
           ("44444448", "lut_finn", 16667, 100.0, "B2 margin 1"))
ALL_LUTRAM = 1 << 30


def tag(d):
    return f"{d[0]}_{d[1]}_{d[2]}"


def memstream_top(depth, width, style):
    """Top for the weight streamer: AXI-lite tied off as in FINN's block design."""
    w = ((width + 7) // 8) * 8
    return f"""module wstrm_top(input ap_clk, input ap_rst_n, input m_axis_0_tready,
  output m_axis_0_tvalid, output [{w - 1}:0] m_axis_0_tdata);
  memstream_axi_wrapper #(.DEPTH({depth}), .WIDTH({width}), .INIT_FILE("memblock.dat"),
    .RAM_STYLE("{style}")) inst (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
    .awvalid(1'b0), .awprot(3'b0), .awaddr(32'b0), .wvalid(1'b0), .wdata(32'b0), .wstrb(4'b0),
    .bready(1'b0), .arvalid(1'b0), .arprot(3'b0), .araddr(32'b0), .rready(1'b0),
    .m_axis_0_tready(m_axis_0_tready), .m_axis_0_tvalid(m_axis_0_tvalid),
    .m_axis_0_tdata(m_axis_0_tdata));
endmodule
"""


def cmd_generate(args):
    import numpy as np  # noqa: F401
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    import copy
    import cost_tables as ct
    from assemble_costs import assemble, choose_variants, load_tables, options
    from energy_search import DERATE, DEVICE, RES, UTIL_W
    table, _ = load_tables(args.tables)
    budget = {k: DEVICE[k] * DERATE[k] for k in RES}
    memdir = os.path.join(args.finn_root, "finn-rtllib", "memstream", "hdl")
    # optional: other designs, as [bits, mode, T, f_MHz, role, {op_type: {attr: value}}] with the
    # attributes forced on every node of that op type (e.g. FINN's default flow with LUT multipliers)
    designs = [tuple(d) for d in json.load(open(args.designs_json))] if args.designs_json else DESIGNS
    cases = []
    for di, d in enumerate(designs):
        bits, mode, T, f, role = d[:5]
        forced = dict(d[5]) if len(d) > 5 else {}
        # "_min_lut": the fewest-LUT variants with no LUT budget (to synthesize a design that the
        # cost model puts just over the budget)
        min_lut = forced.pop("_min_lut", False)
        nodes = assemble(table, list(map(int, bits)), mode, T)
        if mode == "lut_finn":
            chosen = [n["auto"] for n in nodes]
        elif min_lut:
            pick, tot = choose_variants(nodes, dict(budget, LUT=1e9), {"LUT": 1.0, "BRAM_18K": 1e-6, "DSP": 1e-6})
            chosen = [options(n)[p] for n, p in zip(nodes, pick)]
        else:
            pick, tot = choose_variants(nodes, budget, UTIL_W)
            chosen = [options(n)[p] for n, p in zip(nodes, pick)]
        standalone, wwidth = ct.MODES[mode]
        m0 = ct.hw_model(list(map(int, bits)), standalone, tempfile.mkdtemp())
        m = ModelWrapper(copy.deepcopy(m0.model))
        m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
        if mode in ct.DSP_AWARE:
            m = ct.dsp_aware_refold(m, mode in ct.DSP_CORRECTED)
        if args.ts_refold and mode != "lut_finn":   # as cost_tables.py --ts-refold built the table
            if args.ts_hls_simd:
                os.environ["COST_TS_HLS_SIMD"] = str(args.ts_hls_simd)
            m = ct.timing_safe_refold(m, args.ts_rtl_simd, mode in ct.DSP_CORRECTED)
        m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
        m = m.transform(InferDataTypes())
        groups = ct.assign_groups(m)
        hw = sorted(ct.hw_nodes(m), key=lambda n: groups[n.name])  # stable: graph order per group
        assert [n.op_type for n in hw] == [r["op"] for r in nodes], tag(d)
        for i, (n, rec, opt) in enumerate(zip(hw, nodes, chosen)):
            inst = getCustomOp(n)
            assert int(inst.get_exp_cycles()) == rec["cycles"], (tag(d), n.name)
            for k, v in opt.get("attrs", {}).items():
                inst.set_nodeattr(k, ALL_LUTRAM if k == "depth_trigger_bram" and v == 0 else v)
            for k, v in forced.get(n.op_type, {}).items():
                inst.set_nodeattr(k, v)
            name = f"d{di}_{i:02d}_{n.name}"  # short: Vivado inside HLS export hits 260 characters
            cdir = os.path.join(args.out, name)
            shutil.rmtree(cdir, ignore_errors=True)
            os.makedirs(cdir)
            inst.set_nodeattr("code_gen_dir_ipgen", os.path.abspath(cdir))
            kind = "rtl" if n.op_type.endswith("_rtl") else "hls"
            if kind == "rtl":
                inst.generate_hdl(m, PART, 1e3 / f)
            else:
                inst.code_generation_ipgen(m, PART, 1e3 / f)
            pred = {k: float(opt.get(k, 0.0)) for k in RES}
            cases.append({"case": name, "design": tag(d), "role": role, "op": n.op_type,
                          "kind": kind, "f_MHz": f, "predicted": pred,
                          "attrs": {k: str(v) for k, v in {**opt.get("attrs", {}),
                                                           **forced.get(n.op_type, {})}.items()}})
            if n.op_type.startswith("MVAU") and inst.get_nodeattr("mem_mode") == "internal_decoupled":
                wdir = cdir + "_wstrm"
                shutil.rmtree(wdir, ignore_errors=True)
                os.makedirs(wdir)
                for s in glob.glob(os.path.join(memdir, "*")):
                    shutil.copy(s, wdir)
                shutil.copy(os.path.join(cdir, "memblock.dat"), wdir)
                style = inst.get_nodeattr("ram_style")
                open(os.path.join(wdir, "wstrm_top.v"), "w").write(
                    memstream_top(inst.calc_wmem(), inst.get_weightstream_width_padded(), style))
                cases.append({"case": name + "_wstrm", "design": tag(d), "role": role,
                              "op": n.op_type + "_wstrm", "kind": "wstrm", "f_MHz": f,
                              "predicted": {k: 0.0 for k in RES}, "attrs": {"ram_style": style}})
        print(tag(d), role, len(hw), "nodes", flush=True)
    json.dump(cases, open(os.path.join(args.out, "cases.json"), "w"), indent=1)
    print(len(cases), "cases")


def parse_power(text):
    m = re.search(r"^\|\s*Dynamic \(W\)\s*\|\s*([\d.]+)", text, re.M)
    return float(m.group(1)) if m else None


def vivado_power_tcl(f_MHz):
    return (f"create_clock -period {1e3 / f_MHz:.3f} -name ap_clk [get_ports ap_clk]\n"
            "report_power -file vivado_power.rpt\n")


def run_vivado(c, args):
    """RTL node or weight streamer: synthesis, utilisation and power in one Vivado run."""
    import audit_rtl_ops as a
    src = os.path.abspath(os.path.join(args.out, c["case"]))
    d = os.path.join(args.workdir, c["case"])
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".Xil", "*.log", "*.rpt"))
    if c["kind"] == "wstrm":
        srcs = sorted(glob.glob(os.path.join(d, "*.sv")), key=lambda s: (not s.endswith("_pkg.sv"), s))
        srcs += sorted(glob.glob(os.path.join(d, "*.v")))
        srcs = [s.replace("\\", "/") for s in srcs]
        top = "wstrm_top"
    else:
        srcs = a.case_sources(d, c["op"], args.mvu_rtl)
        top = a.top_module(d, c["op"])
    timed = getattr(args, "timing", False)
    if timed:   # the clock is known to synthesis (timing-driven, as in FINN's own build)
        open(os.path.join(d, "clock.xdc"), "w").write(
            f"create_clock -period {1e3 / c['f_MHz']:.3f} -name ap_clk [get_ports ap_clk]\n")
    with open(os.path.join(d, "vivado_synth.tcl"), "w") as f:
        for s in srcs:
            f.write(f"read_verilog {'-sv ' if s.endswith('.sv') else ''}{{{s}}}\n")
        if timed:
            f.write("read_xdc -mode out_of_context clock.xdc\n")
        f.write(f"synth_design -top {top} -part {PART} -mode out_of_context\n")
        f.write("report_utilization -file vivado_util.rpt\n")
        if timed:
            f.write("report_timing_summary -file vivado_timing.rpt\nreport_power -file vivado_power.rpt\n")
        else:
            f.write(vivado_power_tcl(c["f_MHz"]))
    r = subprocess.run([args.vivado, "-mode", "batch", "-nojournal", "-log", "vivado.log",
                        "-source", "vivado_synth.tcl"], cwd=d, capture_output=True, text=True)
    v = parse_reports(d, src, r.returncode, "vivado_util.rpt")
    if timed:
        from rtl_timing import wns
        w = wns(os.path.join(d, "vivado_timing.rpt"))
        v["period_ns"] = None if w is None else round(1e3 / c["f_MHz"] - w, 3)
        if os.path.exists(os.path.join(d, "vivado_timing.rpt")):
            shutil.copy(os.path.join(d, "vivado_timing.rpt"), src)
    return v


def run_hls(c, args):
    """HLS node: Vitis HLS synthesis (audit_hls_thresholds.run_case), then power on its checkpoint."""
    import audit_hls_thresholds as h
    v = h.run_case(c, args)
    d = os.path.join(args.workdir, c["case"])
    dcp = glob.glob(os.path.join(d, "project_*", "sol1", "impl", "verilog", "project.runs",
                                 "bd_0_hls_inst_0_synth_1", "*.dcp"))
    if dcp:
        with open(os.path.join(d, "vivado_power.tcl"), "w") as f:
            f.write(f"open_checkpoint {{{dcp[0].replace(chr(92), '/')}}}\n" + vivado_power_tcl(c["f_MHz"]))
        subprocess.run([args.vivado, "-mode", "batch", "-nojournal", "-log", "vivado_power.log",
                        "-source", "vivado_power.tcl"], cwd=d, capture_output=True, text=True)
    p = os.path.join(d, "vivado_power.rpt")
    v["power_W"] = parse_power(open(p).read()) if os.path.exists(p) else None
    if os.path.exists(p):
        shutil.copy(p, os.path.abspath(os.path.join(args.out, c["case"])))
    src = os.path.abspath(os.path.join(args.out, c["case"]))
    for f in glob.glob(os.path.join(d, "project_*", "sol1", "**", "*_timing_synth.rpt"), recursive=True):
        shutil.copy(f, src)   # keep Vivado's own timing report with the case (the workdir is reused)
    e = os.path.join(src, "export_syn.rpt")   # Vitis HLS: period = target - WNS of Vivado's synthesis
    if os.path.exists(e):
        t = open(e, encoding="utf-8", errors="ignore").read()
        t = t[t.find("RTL Synthesis Timing Summary"):][:700]
        m = re.search(r"Post-(?:Route|Synthesis)\s*\|\s*([\d.]+)", t)
        v["period_ns"] = float(m.group(1)) if m else None
    return v


def parse_reports(d, src, rc, util_name):
    text = open(os.path.join(d, util_name)).read() if os.path.exists(os.path.join(d, util_name)) else ""

    def cell(name):
        m = re.search(r"^\|\s*" + name + r"\*?\s*\|\s*([\d.]+)", text, re.M)
        return float(m.group(1)) if m else 0.0
    p = os.path.join(d, "vivado_power.rpt")
    for f in (util_name, "vivado_power.rpt", "vivado.log"):
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), src)
    return {"returncode": rc if text else -1, "LUT": cell("Slice LUTs"), "LUT_mem": cell("LUT as Memory"),
            "FF": cell("Slice Registers"), "BRAM18": 2 * cell("Block RAM Tile"), "DSP": cell("DSPs"),
            "power_W": parse_power(open(p).read()) if os.path.exists(p) else None}


def code_hash(args, c):
    """Identical generated code (up to the case's own name and path) is identical hardware."""
    import hashlib
    d = os.path.abspath(os.path.join(args.out, c["case"]))
    node = c["case"].split("_", 2)[2].replace("_wstrm", "")
    h = hashlib.sha1(c["kind"].encode())
    for f in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.isfile(f) and not f.endswith((".log", ".rpt", ".npy")):
            text = open(f, "rb").read().replace(d.encode(), b"<D>").replace(d.replace("\\", "/").encode(), b"<D>")
            h.update(os.path.basename(f).replace(node, "<N>").encode() + text.replace(node.encode(), b"<N>"))
    return h.hexdigest()


def cmd_run(args):
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(args.out, "cases.json")
    cases = json.load(open(path))
    done_path = os.path.join(args.out, "results.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else {}
    todo = [c for c in cases if c["case"] not in done or done[c["case"]].get("returncode") != 0]
    if args.only:
        todo = [c for c in todo if any(s in c["case"] for s in args.only)]
    first, dup = {}, {}
    for c in todo:
        k = code_hash(args, c)
        if k in first:
            dup[c["case"]] = first[k]
        else:
            first[k] = c["case"]
    unique = [c for c in todo if c["case"] not in dup]
    print(f"{len(todo)} cases, {len(unique)} distinct", flush=True)

    def one(c):
        return run_hls(c, args) if c["kind"] == "hls" else run_vivado(c, args)
    with ThreadPoolExecutor(args.jobs) as ex:
        for c, v in zip(unique, ex.map(one, unique)):
            done[c["case"]] = v
            for k, src in dup.items():
                if src == c["case"]:
                    done[k] = dict(v, same_as=src)
            json.dump(done, open(done_path, "w"), indent=1)
            print(f"{c['case']:48s} rc={v.get('returncode')} LUT {v.get('LUT')} BRAM18 {v.get('BRAM18')} "
                  f"DSP {v.get('DSP')} P {v.get('power_W')} T {v.get('period_ns')} ns | pred LUT {c['predicted']['LUT']:.0f} "
                  f"BRAM18 {c['predicted']['BRAM_18K']:.0f} DSP {c['predicted']['DSP']:.0f}", flush=True)


def cmd_report(args):
    cases = json.load(open(os.path.join(args.out, "cases.json")))
    res = json.load(open(os.path.join(args.out, "results.json")))
    designs = [tuple(d) for d in json.load(open(args.designs_json))] if args.designs_json else DESIGNS
    for d in designs:
        cs = [c for c in cases if c["design"] == tag(d)]
        ok = [c for c in cs if res.get(c["case"], {}).get("returncode") == 0]
        pred = {k: sum(c["predicted"][k] for c in cs) for k in ("LUT", "BRAM_18K", "DSP")}
        syn = {k: sum(res[c["case"]].get(k) or 0 for c in ok) for k in ("LUT", "BRAM18", "DSP")}
        missing = [c["case"] for c in cs if c not in ok]
        pw = sum(res[c["case"]].get("power_W") or 0 for c in ok)
        per = [(res[c["case"]]["period_ns"], c["case"]) for c in ok if res[c["case"]].get("period_ns")]
        worst = max(per) if per else None
        print(f"== {tag(d)} ({d[4]}): {len(ok)}/{len(cs)} synthesized | LUT pred {pred['LUT']:.0f} synth "
              f"{syn['LUT']:.0f} | BRAM18 pred {pred['BRAM_18K']:.0f} synth {syn['BRAM18']:.0f} | DSP pred "
              f"{pred['DSP']:.0f} synth {syn['DSP']:.0f} | dynamic power {pw:.3f} W at {d[3]:.0f} MHz"
              + (f" | slowest node {worst[0]:.3f} ns ({worst[1]}, {len(per)} timed)" if worst else "")
              + (f" | not synthesized: {missing}" if missing else ""))
        ops = sorted({c["op"] for c in cs})
        for op in ops:
            o = [c for c in cs if c["op"] == op]
            oo = [c for c in o if c in ok]
            print(f"   {op:34s} n={len(o):2d} LUT pred {sum(c['predicted']['LUT'] for c in o):7.0f} synth "
                  f"{sum(res[c['case']]['LUT'] or 0 for c in oo):7.0f} | BRAM18 pred "
                  f"{sum(c['predicted']['BRAM_18K'] for c in o):5.0f} synth {sum(res[c['case']]['BRAM18'] for c in oo):5.0f}"
                  f" | DSP {sum(c['predicted']['DSP'] for c in o):4.0f} / {sum(res[c['case']]['DSP'] for c in oo):4.0f}"
                  f" | P {sum(res[c['case']].get('power_W') or 0 for c in oo):.3f} W")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["generate", "run", "report"])
    p.add_argument("--tables")
    p.add_argument("--out", required=True)
    p.add_argument("--finn-root")
    p.add_argument("--vivado")
    p.add_argument("--vitis-hls")
    p.add_argument("--mvu-rtl")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--workdir", default="D:/vw")
    p.add_argument("--designs-json", help="design list replacing DESIGNS (see cmd_generate)")
    p.add_argument("--ts-refold", action="store_true",
                   help="generate: timing-safe refold, as cost_tables.py --ts-refold built the table")
    p.add_argument("--ts-rtl-simd", type=int, default=None, help="generate: with --ts-refold")
    p.add_argument("--ts-hls-simd", type=int, default=None,
                   help="generate: with --ts-refold, the folded MVAU_hls cap (cost_tables.py --ts-hls-simd)")
    p.add_argument("--timing", action="store_true",
                   help="run: give RTL synthesis the clock and report its timing")
    p.add_argument("--only", nargs="+", default=None, help="run: only cases whose name contains one of these")
    args = p.parse_args()
    {"generate": cmd_generate, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
