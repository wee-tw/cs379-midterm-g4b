"""Small sweep: export ResNet-8 at a few bit configs, run the FINN estimate flow
at several target FPS values, and collect FINN's estimated resources/throughput."""
import csv
import json
import os
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(S, "venv_finn010", "Scripts", "python.exe")
CODE = r"D:\115-1\embedded sys midterm report\code"
FINN_SRC = (r"C:\Users\wee\AppData\Local\Temp\claude\C--SPB-Data--claude-spares-spare-5cUE7w"
            r"\0ab647fd-9b9f-46c9-8993-886b1139c5f2\scratchpad\repos\finn-v0.10.1")
WORK = r"C:\Users\wee\AppData\Local\Temp\fb\sweep"

CONFIGS = {
    "W8A8": [8] * 8,
    "W4A4": [4] * 8,
    "W2A2": [2] * 8,
    "mix_8-4-2-8": [8, 8, 4, 4, 2, 2, 8, 4],
    "mix_4-4-2-8": [4, 4, 4, 4, 2, 2, 8, 8],
}
ST = os.environ.get("ST") == "1"
FPS = [int(x) for x in sys.argv[1:]] or [1000, 5000, 20000, 100000]

env = dict(os.environ, PYTHONPATH=os.path.join(FINN_SRC, "src") + ";" + CODE,
           FINN_ROOT=FINN_SRC)
os.makedirs(WORK, exist_ok=True)
rows = []
for name, bits in CONFIGS.items():
    onnx_path = os.path.join(WORK, name + ".onnx")
    if not os.path.exists(onnx_path):
        subprocess.run([PY, os.path.join(CODE, "verify_resnet8.py"), "export", "--bits",
                        *map(str, bits), "--out", onnx_path], env=env, check=True,
                       capture_output=True)
    for fps in FPS:
        out = os.path.join(WORK, f"{name}_fps{fps}" + ("_st" if ST else ""))
        env["FINN_BUILD_DIR"] = out + "_tmp"
        os.makedirs(env["FINN_BUILD_DIR"], exist_ok=True)
        r = subprocess.run([PY, os.path.join(CODE, "finn_estimate_resnet8.py"), onnx_path,
                            "--out", out, "--target-fps", str(fps)] + (["--standalone-thresholds"] if ST else []),
                           env=env, capture_output=True, text=True)
        try:
            perf = json.load(open(os.path.join(out, "report", "estimate_network_performance.json")))
            res = json.load(open(os.path.join(out, "report", "estimate_layer_resources.json")))["total"]
        except FileNotFoundError:
            print(name, fps, "FAILED", r.stdout[-500:], r.stderr[-1500:])
            continue
        row = dict(config=name + ("+ST" if ST else ""), target_fps=fps, est_fps=round(perf["estimated_throughput_fps"], 1),
                   est_latency_ms=round(perf["estimated_latency_ns"] / 1e6, 3),
                   LUT=int(res["LUT"]), BRAM18=res["BRAM_18K"], DSP=int(res["DSP"]))
        print(row, flush=True)
        rows.append(row)

with open(os.path.join(WORK, "sweep_st.csv" if ST else "sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
