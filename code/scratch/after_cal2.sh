#!/bin/bash
# After after_calib.sh: stack the Vitis HLS calibration of MVAU_hls (calib_mvau_hls.py) and the
# replay-buffer BRAM of MVAU_rtl (replay_bram.py) on the Vivado-calibrated tables, then rerun the
# main pipeline (implementation_notes.md 12.31).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
until grep -q "== done" "$S/after_calib.log" 2>/dev/null; do sleep 60; done
echo "== HLS calibration and replay BRAM on the Vivado tables $(date +%T)"
"$PY" calib_mvau_hls.py fit --audit "$S/audit_hls_finn" "$S/audit_hls"
"$PY" calib_mvau_hls.py tables --audit "$S/audit_hls_finn" "$S/audit_hls" --tables "$S/cost/tables_viv.json" --out "$S/cost/tables_viv_hls.json"
"$PY" replay_bram.py tables --tables "$S/cost/tables_viv_hls.json" --out "$S/cost/tables_cal.json"
T="$S/cost/tables_cal.json"; P="$S/cost/points_cal.npz"; L="$S/cost/latency_cal.npz"
"$PY" energy_search.py points --tables "$T" --out "$P" > "$S/cost/points_cal.log" 2>&1
tail -1 "$S/cost/points_cal.log"
"$PY" latency_check.py points --tables "$T" --points "$P" --out "$L"
"$PY" threshold_sweep.py --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --out "$S/cost/threshold_sweep_cal.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-200
"$PY" robust_select.py --points "$P" --acc "$S/cost/acc_model_3rep.json" | cut -c1-300
echo "== done $(date +%T)"
