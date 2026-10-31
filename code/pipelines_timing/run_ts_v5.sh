#!/bin/bash
# Refold checked by routing (notes 12.41): MVAU_rtl to SIMD <= 12 (125 MHz after routing at PE 4-32)
# and folded MVAU_hls to SIMD <= 18 (111 MHz after routing; SIMD 24 routed at 9.45 ns), then the
# same steps as the main results (run_ts_v3.sh). Usage: run_ts_v5.sh <workers> <rtl timing json>
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"; PY="$S/venv_finn010/Scripts/python.exe"; C="$S/cost"
W="${1:-2}"; RTL="$2"
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
need "$RTL"
cd "$CODE" || exit 1
echo "== cost tables (RTL SIMD <= 12, HLS SIMD <= 18; lut, dspc, dsp4c; $W workers) $(date +%T)"
PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
    "$PY" cost_tables.py --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18 --modes lut,dspc,dsp4c --workers "$W" \
    --out "$C/tables_tsr1218_raw.json" > "$C/tables_tsr1218_raw.log" 2>&1
need "$C/tables_tsr1218_raw.json"; tail -1 "$C/tables_tsr1218_raw.log" | cut -c1-200
bash "$S/run_ts_v3.sh" "$C/tables_tsr1218_raw.json" "v1218" "$RTL"
