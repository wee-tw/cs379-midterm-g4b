#!/bin/bash
# Main pipeline on the second LUT calibration (calib_v2.py: lane-count RTL MVU model, HLS input-buffer
# and weight-width terms, small operators, SWG ratio; implementation_notes.md 12.33).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
C="$S/cost"; T="$C/tables_calv2.json"; P="$C/points_calv2.npz"; L="$C/latency_calv2.npz"; A="$C/acc_model_3rep.json"
echo "== points $(date +%T)"
"$PY" energy_search.py points --tables "$T" --out "$P" > "$C/points_calv2.log" 2>&1
tail -1 "$C/points_calv2.log"
"$PY" latency_check.py points --tables "$T" --points "$P" --out "$L"
echo "== threshold sweep $(date +%T)"
"$PY" threshold_sweep.py --points "$P" --acc "$A" --latency "$L" --out "$C/threshold_sweep_calv2.json" \
    | grep -E "^== margin|cont vs|jit:" | cut -c1-200
"$PY" robust_select.py --points "$P" --acc "$A" | cut -c1-300
echo "== budget sweep $(date +%T)"
"$PY" budget_sweep.py --tables "$T" --acc "$A" --out "$C/budget_sweep_calv2.json" > "$C/budget_sweep_calv2.log" 2>&1
grep -E "nominal|^LUT" "$C/budget_sweep_calv2.log" | head -12
echo "== latency-aware idle $(date +%T)"
"$PY" energy_search.py idle --points "$P" --acc "$A" --latency "$L" --lat-scale 0.6024 \
    --out "$C/idle_lat_calv2.json" > "$C/idle_lat_calv2.log" 2>&1
"$PY" latency_summary.py "$C/idle_lat_calv2.json" | grep -E "^== nominal|^== z2|R=  300|R= 3000" | cut -c1-240
echo "== done $(date +%T)"
