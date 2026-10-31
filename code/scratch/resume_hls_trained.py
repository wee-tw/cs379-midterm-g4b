"""Finish the trained-weight HLS audit after its process was killed: take the cases whose
synthesis report already exists in the work dir, run the rest, write audit_hls.json."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "D:/115-1/embedded sys midterm report/code")
import audit_hls_thresholds as h  # noqa: E402

S = "C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
args = argparse.Namespace(out=f"{S}/audit_hls_trained", workdir="D:/ht", vitis_hls="D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat",
                          finn_root="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1")
cases = json.load(open(f"{args.out}/cases.json"))
todo = []
for c in cases:
    u = h.parse_util(os.path.join(args.workdir, c["case"]))
    if u:
        c["synth"] = {"returncode": 0, **u}
    else:
        todo.append(c)
print(len(cases) - len(todo), "already synthesized,", len(todo), "to run", flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
    for c, v in zip(todo, ex.map(lambda c: h.run_case(c, args), todo)):
        c["synth"] = v
        print(f"{c['case']:32s} rc={v['returncode']} LUT finn {c['finn_estimate']['LUT']:7.0f} synth {v.get('LUT')} | "
              f"BRAM18 synth {v.get('BRAM18')}", flush=True)
json.dump(cases, open(f"{args.out}/audit_hls.json", "w"), indent=1)
print("wrote audit_hls.json", flush=True)
