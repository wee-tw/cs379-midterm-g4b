"""Rerun the SWG (ConvolutionInputGenerator_rtl) cases of the Vivado audit after the source-order fix.
Reads cases.json (not audit.json, which calib_mvu_vivado.py prepare may be rewriting) and writes
swg_vivado.json next to it; merge into audit.json afterwards."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "D:/115-1/embedded sys midterm report/code")
import audit_rtl_ops as a  # noqa: E402

S = "C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
args = argparse.Namespace(out=f"{S}/audit", workdir="D:/vs", mvu_rtl=f"{S}/upstream/mvu_v0.10.1",
                          vivado="D:/Xilinx/Vivado/2022.2/bin/vivado.bat")
cases = [c for c in json.load(open(f"{S}/audit/cases.json")) if c["op"] == "ConvolutionInputGenerator_rtl"]
res = {}
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as ex:
    for c, v in zip(cases, ex.map(lambda c: a.vivado_case(c, args), cases)):
        res[c["case"]] = v
        print(f"{c['case']:34s} rc={v['returncode']} LUT {v['LUT']:6.0f} (mem {v['LUT_mem']:4.0f}) "
              f"FF {v['FF']:5.0f} BRAM18 {v['BRAM18']:4.1f} DSP {v['DSP']:3.0f}", flush=True)
json.dump(res, open(f"{S}/audit/swg_vivado.json", "w"), indent=1)
print("ok", sum(v["returncode"] == 0 for v in res.values()), "of", len(res))
