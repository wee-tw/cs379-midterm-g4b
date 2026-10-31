#!/bin/bash
# Narrower RTL refold (notes 12.41). Usage: run_ts_v4.sh <rtl simd cap> <workers> <rtl timing json>
# 1. cost_tables.py --ts-refold --ts-rtl-simd CAP for the DSP modes (the lut mode has only SIMD-1
#    RTL MVUs, so its refolded records come unchanged from tables_tsr2_raw.json)
# 2. merge into one raw table, 3. run_ts_v3.sh: calibration chain, classes, points, merged sweep
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"; PY="$S/venv_finn010/Scripts/python.exe"; C="$S/cost"
CAP="$1"; W="${2:-2}"; RTL="$3"
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
need "$RTL"; need "$C/tables_tsr2_raw.json"
cd "$CODE" || exit 1
echo "== cost tables (RTL SIMD <= $CAP, dspc + dsp4c, $W workers) $(date +%T)"
PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    "$PY" cost_tables.py --ts-refold --ts-rtl-simd "$CAP" --modes dspc,dsp4c --workers "$W" \
    --out "$C/tables_tsr${CAP}_raw_dsp.json" > "$C/tables_tsr${CAP}_raw_dsp.log" 2>&1
need "$C/tables_tsr${CAP}_raw_dsp.json"; tail -1 "$C/tables_tsr${CAP}_raw_dsp.log" | cut -c1-200
echo "== merge with the refolded lut records $(date +%T)"
"$PY" - "$C/tables_tsr2_raw.json" "$C/tables_tsr${CAP}_raw_dsp.json" "$C/tables_tsr${CAP}_raw.json" <<'PYEOF'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
t = {k: v for k, v in a["table"].items() if k.startswith("lut|")}
assert all(k.split("|")[0] in ("dspc", "dsp4c") for k in b["table"])
t.update(b["table"])
assert len(t) == len(a["table"]), (len(t), len(a["table"]))
json.dump({**a, "table": t, "mismatches": a["mismatches"] + b["mismatches"]}, open(sys.argv[3], "w"))
print("records", len(t))
PYEOF
need "$C/tables_tsr${CAP}_raw.json"
bash "$S/run_ts_v3.sh" "$C/tables_tsr${CAP}_raw.json" "v$CAP" "$RTL"
