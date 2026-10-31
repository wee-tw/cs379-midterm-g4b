#!/bin/bash
# Sensitivity of the calibrated main results to a +-15% systematic error in the RTL-MVU or
# HLS-MVU LUT calibration (calib_sensitivity.py; implementation_notes.md 12.31). Two runs at a
# time to leave memory for the Vivado / HLS jobs.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
C="$S/cost"
run() {  # name rtl hls
  "$PY" calib_sensitivity.py tables --tables "$C/tables_calov.json" --rtl $2 --hls $3 --out "$C/tables_sens_$1.json" | tail -1
  "$PY" energy_search.py points --tables "$C/tables_sens_$1.json" --out "$C/points_sens_$1.npz" > "$C/points_sens_$1.log" 2>&1
  "$PY" threshold_sweep.py --points "$C/points_sens_$1.npz" --acc "$C/acc_model_3rep.json" \
      --out "$C/threshold_sweep_sens_$1.json" > /dev/null 2>&1
  rm -f "$C/tables_sens_$1.json"
  echo "$1 done $(date +%T)"
}
echo "== start $(date +%T)"
run rtl085 0.85 1.0 & run rtl115 1.15 1.0 & wait
run hls085 1.0 0.85 & run hls115 1.0 1.15 & wait
"$PY" calib_sensitivity.py summary --base "$C/threshold_sweep_calov.json" --runs \
  "RTL MVU LUT x0.85=$C/threshold_sweep_sens_rtl085.json" "RTL MVU LUT x1.15=$C/threshold_sweep_sens_rtl115.json" \
  "HLS MVU LUT x0.85=$C/threshold_sweep_sens_hls085.json" "HLS MVU LUT x1.15=$C/threshold_sweep_sens_hls115.json"
echo "== done $(date +%T)"
