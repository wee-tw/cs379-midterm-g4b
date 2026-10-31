#!/bin/bash
# After the Vivado audit (after_vivado.sh): (1) prove FINN's chosen designs do not fit,
# (2) calibrate the RTL-MVU LUT cost on Vivado and rerun the main pipeline on it.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
until grep -q "== done" "$S/after_vivado.log" 2>/dev/null; do sleep 60; done
echo "== infeasible designs through Vivado $(date +%T)"
"$PY" prove_infeasible.py vivado --out "$S/infeasible" --vivado "D:/Xilinx/Vivado/2022.2/bin/vivado.bat" \
    --mvu-rtl "$S/upstream/mvu_v0.10.1" --jobs 4 2>&1 | tail -8
echo "== MVU LUT calibration $(date +%T)"
"$PY" calib_mvu_vivado.py prepare --audit "$S/audit" 2>&1 | grep -v "outside fpgadataflow" | tail -2
"$PY" calib_mvu_vivado.py fit --audit "$S/audit"
"$PY" calib_mvu_vivado.py tables --audit "$S/audit" --tables "$S/cost/tables_thr_hi.json" --out "$S/cost/tables_viv.json"
T="$S/cost/tables_viv.json"; P="$S/cost/points_viv.npz"; L="$S/cost/latency_viv.npz"
"$PY" energy_search.py points --tables "$T" --out "$P" > "$S/cost/points_viv.log" 2>&1
tail -1 "$S/cost/points_viv.log"
"$PY" latency_check.py points --tables "$T" --points "$P" --out "$L"
"$PY" threshold_sweep.py --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --out "$S/cost/threshold_sweep_viv.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-200
"$PY" robust_select.py --points "$P" --acc "$S/cost/acc_model_3rep.json" | cut -c1-300
echo "== done $(date +%T)"
