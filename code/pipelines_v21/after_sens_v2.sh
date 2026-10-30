#!/bin/bash
# Sensitivity of the v2 main results to a design-level LUT error (every node x0.90/0.95/1.05/1.10;
# the leave-one-design-out error of calib_v2 is within +-5%). Waits for the harm study (memory).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
C="$S/cost"
run() {  # name all-scale
  "$PY" calib_sensitivity.py tables --tables "$C/tables_calv2.json" --all $2 --out "$C/tables_sv2_$1.json" | tail -1
  "$PY" energy_search.py points --tables "$C/tables_sv2_$1.json" --out "$C/points_sv2_$1.npz" > "$C/points_sv2_$1.log" 2>&1
  "$PY" threshold_sweep.py --points "$C/points_sv2_$1.npz" --acc "$C/acc_model_3rep.json" \
      --out "$C/threshold_sweep_sv2_$1.json" > /dev/null 2>&1
  rm -f "$C/tables_sv2_$1.json"
  echo "$1 done $(date +%T)"
}
echo "== start $(date +%T)"
run all095 0.95 & run all105 1.05 & wait
run all090 0.90 & run all110 1.10 & wait
"$PY" calib_sensitivity.py summary --base "$C/threshold_sweep_calv2.json" --runs \
  "all LUT x0.90=$C/threshold_sweep_sv2_all090.json" "all LUT x0.95=$C/threshold_sweep_sv2_all095.json" \
  "all LUT x1.05=$C/threshold_sweep_sv2_all105.json" "all LUT x1.10=$C/threshold_sweep_sv2_all110.json"
echo "== done $(date +%T)"
