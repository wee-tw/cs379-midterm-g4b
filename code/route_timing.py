"""Post-route timing of single nodes (implementation_notes.md 12.40).

The clock classes (timing_safe.py) come from post-synthesis timing, which is optimistic. This
places and routes single nodes out of context (Vivado 2022.2, xc7z020clg400-1, the steps of
the Vivado Implementation Defaults strategy: opt_design, place_design, phys_opt_design,
route_design) and compares the achieved period after synthesis and after routing:
  RTL nodes   the case's synthesis script (verify_designs.py vivado_synth.tcl, or rtl_timing.py
              --constrain-synth timing.tcl: sources, the clock read before synthesis), then
              implementation
  HLS nodes   the checkpoint of Vitis HLS's own Vivado synthesis (export_design -flow syn),
              then the same steps
The ports become partition pins (HD.PARTITION), so wide streams do not need I/O pins. A single
node in an empty device routes more easily than inside a full design, so the post-route
numbers are still a lower bound on the period.

  python route_timing.py --workdir D:/vw5 --cases d0_19_MVAU_rtl_3 d0_02_MVAU_rtl_0 \
      --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat --out-dir D:/vr --out route_timing.json
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess

from rtl_timing import wns

IMPL = ("set_property HD.PARTITION 1 [current_design]\n"
        "report_timing_summary -file synth_timing.rpt\n"
        "opt_design\nplace_design\nphys_opt_design\nroute_design\n"
        "report_timing_summary -file route_timing.rpt\n"
        "report_utilization -file route_util.rpt\n")


def period(rpt):
    """Achieved period (ns) from a report_timing_summary file: the clock's period minus WNS."""
    w = wns(rpt)
    if w is None:
        return None
    text = open(rpt, encoding="utf-8", errors="replace").read()
    i = text.find("Clock Summary")
    for line in text[i:].splitlines()[4:12]:
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ("ap_clk", "clk"):
            return round(float(parts[-2]) - w, 3)
    return None


def run_case(case, args):
    src = os.path.join(args.workdir, case)
    d = os.path.join(args.out_dir, case)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    # RTL node: the synthesis script of verify_designs.py (vivado_synth.tcl) or rtl_timing.py
    # (timing.tcl, with --constrain-synth); same sources and clock, synthesis then implementation
    synth = next((os.path.join(src, f) for f in ("vivado_synth.tcl", "timing.tcl")
                  if os.path.exists(os.path.join(src, f))), None)
    if synth:
        for f in glob.glob(os.path.join(src, "*")):
            if os.path.isfile(f) and not f.endswith((".log", ".rpt", ".jou")):
                shutil.copy(f, d)
        if args.period:   # same constraint for every node: synthesis and implementation at it
            xdc = open(os.path.join(d, "clock.xdc")).read()
            with open(os.path.join(d, "clock.xdc"), "w") as f:
                f.write(re.sub(r"-period [\d.]+", f"-period {args.period:.3f}", xdc))
        lines = open(synth).read().splitlines()
        head = [ln for ln in lines if ln.startswith(("read_", "synth_design"))]
        tcl = "\n".join(head).replace(src.replace("\\", "/"), d.replace("\\", "/")) + "\n"
        kind = "rtl"
    else:   # HLS node: Vitis HLS's synthesis checkpoint (synthesized at its HLS clock)
        dcp = glob.glob(os.path.join(src, "project_*", "sol1", "impl", "verilog", "project.runs",
                                     "bd_0_hls_inst_0_synth_1", "*.dcp"))
        if not dcp:
            return {"case": case, "error": "no synthesis checkpoint"}
        tcl = f"open_checkpoint {{{dcp[0].replace(chr(92), '/')}}}\n"
        if args.period:   # implementation at the common constraint (replaces the HLS clock)
            tcl += f"create_clock -period {args.period:.3f} -name ap_clk [get_ports ap_clk]\n"
        kind = "hls"
    with open(os.path.join(d, "route.tcl"), "w") as f:
        f.write(tcl + IMPL)
    r = subprocess.run([args.vivado, "-mode", "batch", "-nojournal", "-log", "route.log",
                        "-source", "route.tcl"], cwd=d, capture_output=True, text=True)
    s, p = period(os.path.join(d, "synth_timing.rpt")), period(os.path.join(d, "route_timing.rpt"))
    return {"case": case, "kind": kind, "returncode": r.returncode, "synth_ns": s, "route_ns": p,
            "route_over_synth": None if not (s and p) else round(p / s - 1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="verify_designs.py run --workdir")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--vivado", required=True)
    ap.add_argument("--period", type=float, default=None,
                    help="clock period (ns) for every node; default: the case's own clock")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out-dir", default="D:/vr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for res in ex.map(lambda c: run_case(c, args), args.cases):
            out.append(res)
            print(json.dumps(res), flush=True)
            json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
