#!/bin/bash
# After run_ts_v5.sh (refold RTL <= 12, HLS <= 18): the main table with measured accuracy
# (timing_safe.py table) and a +10 % routing-margin check of the same tables (run_ts_classes.sh).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
C="$S/cost"; L="D:/115-1/embedded sys midterm report/results/accuracy/train_logs"; PY="$S/venv_finn010/Scripts/python.exe"
RTL="$S/rtl_timing/rtl_all_constrained.json"
CLASSES="125 111.11 100 90.91 83.33 76.92 71.43 66.67 50"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
until grep -q "^== done" "$S/run_ts_v5.log" 2>/dev/null || grep -q "MISSING\|Traceback" "$S/run_ts_v5.log" 2>/dev/null; do sleep 30; done
grep -q "MISSING\|Traceback" "$S/run_ts_v5.log" && { echo "run_ts_v5 failed, stopping"; exit 1; }
cd "D:/115-1/embedded sys midterm report/code" || exit 1
table() {   # table <points tag> <sweep json> <out tag>
  PTS=""; for c in $CLASSES; do PTS="$PTS $c=$C/points_$1_f$c.npz"; done
  "$PY" timing_safe.py table --acc "$C/acc_model_3rep.json" --sweep "$2" --points $PTS --logs "$L"/exp* \
      --extra 44444444=83.94,83.83 --out "$C/main_table_$3.json" > "$C/main_table_$3.log" 2>&1
  echo "== main table $3 $(date +%T)"; grep -E "^== |margin|B1_measured|B2_measured" "$C/main_table_$3.log" | head -40
}
table tsr_v1218 "$C/threshold_sweep_tsr_v1218_merged.json" tsr_v1218_m0
echo "== routing margin +10% $(date +%T)"
bash "$S/run_ts_classes.sh" "$C/tables_calv2_tsr_v1218.json" tsr_v1218_m0.1 "$RTL" 3 0.1 | grep -E "records by class|^== margin|nominal|MISSING|done"
table tsr_v1218_m0.1 "$C/threshold_sweep_tsr_v1218_m0.1_merged.json" tsr_v1218_m0.1
echo "== all done $(date +%T)"
