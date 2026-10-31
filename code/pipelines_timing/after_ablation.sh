#!/bin/bash
# Ablation of the timing-safe refold under the same routing-checked class model (notes 12.43):
#   no refold (tables_calv2.json)  vs  refold SIMD <= 27 (tsr_v2)  vs  RTL <= 12 / HLS <= 18 (tsr_v1218)
# Folded MVAU_hls wider than SIMD 18 take the routed 9.45 ns (--hls-routed-simd 18).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
C="$S/cost"; L="D:/115-1/embedded sys midterm report/results/accuracy/train_logs"; PY="$S/venv_finn010/Scripts/python.exe"
RTL="$S/rtl_timing/rtl_all_constrained.json"
CLASSES="125 111.11 100 90.91 83.33 76.92 71.43 66.67 50"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
until grep -q "^== all done" "$S/after_v5.log" 2>/dev/null || grep -q "^run_ts_v5 failed" "$S/after_v5.log" 2>/dev/null; do sleep 30; done
grep -q "^== all done" "$S/after_v5.log" || { echo "after_v5 did not finish, stopping"; exit 1; }
cd "D:/115-1/embedded sys midterm report/code" || exit 1
for pair in "tables_calv2.json calv2c_r" "tables_calv2_tsr_v2.json tsr_v2c_r"; do
  set -- $pair
  echo "######## $2 $(date +%T)"
  bash "$S/run_ts_classes2.sh" "$C/$1" "$2" "$RTL" 3 0 --hls-routed-simd 18 | grep -E "records by class|^== margin|nominal|MISSING|done"
  PTS=""; for c in $CLASSES; do PTS="$PTS $c=$C/points_$2_f$c.npz"; done
  "$PY" timing_safe.py table --acc "$C/acc_model_3rep.json" --sweep "$C/threshold_sweep_$2_merged.json" --points $PTS \
      --logs "$L"/exp* --out "$C/main_table_$2.json" > "$C/main_table_$2.log" 2>&1
  grep -E "^== |margin|B1_measured|B2_measured" "$C/main_table_$2.log" | head -40
done
echo "== ablation done $(date +%T)"
