"""Post-synthesis timing of FINN v0.10.1 RTL operators (implementation_notes.md 12.35).

Out-of-context Vivado synthesis (xc7z020) of audit cases made by audit_rtl_ops.py generate,
with a clock on ap_clk, then report_timing_summary. Post-synthesis timing is optimistic: a
node that misses here will not close after routing; one that meets it still might not.

  python rtl_timing.py --audit <S>/audit --mvu-rtl <S>/upstream/mvu_v0.10.1 \
      --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat --cases MVAU_rtl_146 FMPadding_rtl_126 \
      [--period 8] [--jobs 2] [--workdir D:/vt] --out rtl_timing.json
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess

from audit_rtl_ops import PART, case_sources, top_module


def wns(path):
    """WNS (ns) from the Design Timing Summary of report_timing_summary, or None."""
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line:
            for nxt in lines[i + 1:i + 4]:
                m = re.match(r"\s*(-?\d+\.\d+)\s", nxt)
                if m:
                    return float(m.group(1))
    return None


def run_case(c, args):
    src = os.path.abspath(os.path.join(args.audit, c["case"]))
    d = os.path.join(args.workdir, c["case"])
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".Xil", "*.log", "*.ys", "*.rpt"))
    top = top_module(d, c["op"])
    clock = f"create_clock -name clk -period {args.period:.3f} [get_ports ap_clk]\n"
    if args.constrain_synth:   # clock known to synthesis (timing-driven), as FINN's flow does
        with open(os.path.join(d, "clock.xdc"), "w") as f:
            f.write(clock)
    with open(os.path.join(d, "timing.tcl"), "w") as f:
        for s in case_sources(d, c["op"], args.mvu_rtl):
            f.write(f"read_verilog {'-sv ' if s.endswith('.sv') else ''}{{{s}}}\n")
        if args.constrain_synth:
            f.write("read_xdc -mode out_of_context clock.xdc\n")
        f.write(f"synth_design -top {top} -part {PART} -mode out_of_context\n")
        if not args.constrain_synth:
            f.write("set clks [get_ports -quiet ap_clk]\n")
            f.write("if {[llength $clks] == 0} {set clks [get_ports -quiet *clk*]}\n")
            f.write(f"create_clock -name clk -period {args.period:.3f} $clks\n")
        f.write("report_timing_summary -file timing.rpt\n")
        f.write("report_utilization -file util.rpt\n")
    # cwd = case dir: the threshold RTL reads its $readmemh files by relative path
    r = subprocess.run([args.vivado, "-mode", "batch", "-nojournal", "-log", "vivado_timing.log",
                        "-source", "timing.tcl"], cwd=d, capture_output=True, text=True)
    w = wns(os.path.join(d, "timing.rpt"))
    return {"case": c["case"], "op": c["op"], "attrs": c["attrs"], "returncode": r.returncode,
            "period_ns": args.period, "WNS_ns": w,
            "achieved_ns": None if w is None else round(args.period - w, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--mvu-rtl", required=True)
    ap.add_argument("--vivado", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--period", type=float, default=8.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--workdir", default="D:/vt")
    ap.add_argument("--constrain-synth", action="store_true",
                    help="give synth_design the clock (XDC read before synthesis)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = {c["case"]: c for c in json.load(open(os.path.join(args.audit, "cases.json")))}
    todo = [cases[n] for n in args.cases]
    out = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for res in ex.map(lambda c: run_case(c, args), todo):
            out.append(res)
            a = res["attrs"]
            print(f"{res['case']:36s} rc={res['returncode']} PE/SIMD {a.get('PE')}/{a.get('SIMD')} "
                  f"W {a.get('weightDataType')} in {a.get('inputDataType')} | WNS {res['WNS_ns']} ns "
                  f"at {args.period} ns -> achieved {res['achieved_ns']} ns", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
