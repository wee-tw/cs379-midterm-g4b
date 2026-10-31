"""Check the threshold cost model of threshold_consistent.py (trigger variants) against Vivado.

The audit synthesized FINN's default Thresholding_rtl (both triggers 0, RAM_STYLE "auto"
everywhere). The cost tables instead price explicit triggers: depth_trigger_bram = D puts
stages at least D deep in BRAM and forces stages 64..D-1 deep to distributed RAM; a very large
D realizes the all-LUTRAM variant. This copies every audited Thresholding_rtl case with
DEPTH_TRIGGER_BRAM = 2^30 and = 1024, synthesizes it out of context and compares with the model.
Run with the FINN venv (the model uses FINN's memutil)."""
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import argparse

sys.path.insert(0, "D:/115-1/embedded sys midterm report/code")
import audit_rtl_ops as a  # noqa: E402
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal  # noqa: E402

S = "C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
OUT = f"{S}/audit_thr_trig"
BRAM_PRIMS = {k: v for k, v in mem_primitives_versal.items() if k.startswith("BRAM")}


def model(C, pe, N, wt, D):
    lut = bram = 0
    for d in [C // pe * 2 ** x for x in range(N)]:
        if d >= D:
            bram += pe * get_memutil_alternatives((wt, d), BRAM_PRIMS)[0][1][0]
        else:
            lut += pe * -(-d // 64) * wt
    return lut, bram


cases = [c for c in json.load(open(f"{S}/audit/cases.json")) if c["op"] == "Thresholding_rtl"]
jobs = []
for c in cases:
    src = f"{S}/audit/{c['case']}"
    wrapper = [f for f in os.listdir(src) if f.endswith("_axi_wrapper.v")][0]
    text = open(os.path.join(src, wrapper), encoding="utf-8", errors="replace").read()
    p = {k: int(re.search(rf"parameter\s+{k}\s*=\s*(\d+)", text).group(1)) for k in ("N", "WT", "C", "PE")}
    for D in (1 << 30, 1024):
        name = f"{c['case']}_D{D}"
        d = os.path.join(OUT, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d, ignore=shutil.ignore_patterns("*.log", "*.rpt", "*.ys"))
        open(os.path.join(d, wrapper), "w", encoding="utf-8").write(
            re.sub(r"(parameter\s+DEPTH_TRIGGER_BRAM\s*=\s*)\d+", rf"\g<1>{D}", text))
        jobs.append({"case": name, "op": "Thresholding_rtl", "params": p, "D": D,
                     "model": model(p["C"], p["PE"], p["N"], p["WT"], D)})
args = argparse.Namespace(out=OUT, workdir="D:/vt", mvu_rtl=None,
                          vivado="D:/Xilinx/Vivado/2022.2/bin/vivado.bat")
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as ex:
    for j, v in zip(jobs, ex.map(lambda j: a.vivado_case(j, args), jobs)):
        j["vivado"] = v
        print(f"{j['case']:34s} rc={v['returncode']} model LUT {j['model'][0]:5d} BRAM18 {j['model'][1]:3d} | "
              f"vivado LUT {v['LUT']:5.0f} (mem {v['LUT_mem']:4.0f}) BRAM18 {v['BRAM18']:4.1f}", flush=True)
json.dump(jobs, open(f"{OUT}/thr_trigger.json", "w"), indent=1)
for D in (1 << 30, 1024):
    js = [j for j in jobs if j["D"] == D and j["vivado"]["returncode"] == 0]
    print(f"D={D}: {len(js)} nodes, model LUT {sum(j['model'][0] for j in js)} BRAM18 "
          f"{sum(j['model'][1] for j in js)} | vivado LUT {sum(j['vivado']['LUT'] for j in js):.0f} "
          f"(LUT as memory {sum(j['vivado']['LUT_mem'] for j in js):.0f}) BRAM18 "
          f"{sum(j['vivado']['BRAM18'] for j in js):.0f}; BRAM exact in "
          f"{sum(j['model'][1] == j['vivado']['BRAM18'] for j in js)}")
