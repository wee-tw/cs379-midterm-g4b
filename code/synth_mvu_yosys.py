"""Synthesize FINN-generated MVAU_rtl wrappers for xc7 with Yosys and count cells.

Second check of FINN v0.10.1's MVAU_rtl estimates (implementation_notes.md 12.21), after
the slang elaboration count: the wrapper written by FINN's own generate_hdl() plus
finn-rtllib/mvu is read with the slang frontend (yosys -m slang) and mapped with
synth_xilinx -family xc7. Vendor primitives the RTL instantiates (DSP48E1, LUT6_2) are
kept as cells; everything else is mapped to LUTs, flip-flops and carry chains, so `stat`
gives the DSPs and a LUT count from an actual synthesis (Yosys, not Vivado).

  python synth_mvu_yosys.py --yosys <oss-cad-suite>/bin/yosys.exe --rtl <finn-rtllib/mvu> \
      --cases <dir with one generated wrapper per subdirectory> --out synth.json
"""

import argparse
import glob
import json
import os
import re
import subprocess

CELLS = ("DSP48E1", "LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "LUT6_2", "CARRY4",
         "FDRE", "FDSE", "FDCE", "FDPE", "SRL16E", "SRLC32E", "RAMB18E1", "RAMB36E1",
         "RAM32M", "RAM64M", "MUXF7", "MUXF8")


def run_case(yosys, rtl, prims, wrapper, nodsp, log):
    top = os.path.basename(wrapper).replace("_wrapper.v", "")
    srcs = [prims] + sorted(glob.glob(os.path.join(rtl, "*.sv"))) + [wrapper]
    srcs = [s.replace("\\", "/") for s in srcs]   # yosys scripts: no quotes, forward slashes
    assert not any(" " in s for s in srcs), "paths with spaces need a different quoting"
    ys = log[:-4] + ".ys"
    with open(ys, "w") as f:
        f.write(f"read_slang --top {top} " + " ".join(srcs) + "\n")
        f.write(f"synth_xilinx -family xc7 -top {top} -flatten" + (" -nodsp" if nodsp else "") + "\n")
        f.write("stat\n")
    r = subprocess.run([yosys, "-m", "slang", "-q", "-l", log, "-s", ys],
                       capture_output=True, text=True)
    text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    counts = {}
    # the final `stat` of the script (the flattened top after synth_xilinx)
    block = text[text.rfind("Printing statistics"):] if "Printing statistics" in text else ""
    for c in CELLS:
        m = re.search(rf"^\s+(?:\$?{c})\s+(\d+)\s*$|^\s+(\d+)\s+(?:\$?{c})\s*$", block, re.M)
        if m:
            counts[c] = int(m.group(1) or m.group(2))
    return {"top": top, "returncode": r.returncode, "nodsp": nodsp, "counts": counts,
            "stderr": r.stderr[-800:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yosys", required=True)
    p.add_argument("--rtl", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--prims", required=True,
                   help="(* blackbox *) declarations of DSP48E1/DSP48E2/LUT6_2: sv-elab needs "
                        "a definition for every module")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    # the suite's DLLs live next to yosys (bin) and in lib, as its environment.bat sets up
    root = os.path.dirname(os.path.dirname(os.path.abspath(args.yosys)))
    os.environ["PATH"] = os.pathsep.join([os.path.join(root, "bin"), os.path.join(root, "lib"),
                                          os.environ.get("PATH", "")])
    res = []
    for d in sorted(glob.glob(os.path.join(args.cases, "*"))):
        wrappers = [w for w in glob.glob(os.path.join(d, "*_wrapper.v"))]
        if not wrappers:
            continue
        for nodsp in (True, False):
            log = os.path.join(d, "yosys" + ("_nodsp" if nodsp else "") + ".log")
            r = run_case(args.yosys, args.rtl, args.prims, wrappers[0], nodsp, log)
            r["case"] = os.path.basename(d)
            res.append(r)
            c = r["counts"]
            luts = sum(c.get(k, 0) for k in ("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"))
            print(f"{r['case']:24s} {'nodsp' if nodsp else 'dsp  '} rc={r['returncode']} "
                  f"DSP48E1={c.get('DSP48E1', 0)} LUT6_2={c.get('LUT6_2', 0)} LUT1-6={luts} "
                  f"CARRY4={c.get('CARRY4', 0)} FF={sum(c.get(k, 0) for k in ('FDRE', 'FDSE', 'FDCE', 'FDPE'))} "
                  f"SRL={c.get('SRL16E', 0) + c.get('SRLC32E', 0)}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
