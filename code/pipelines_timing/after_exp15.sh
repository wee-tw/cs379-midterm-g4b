#!/bin/bash
# After exp15 (24222488, 3 seeds): copy the logs to the repo and rebuild the three main tables
# (timing_safe.py table; B2_measured at margin 5 needs these logs).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
C="$S/cost"; L="D:/115-1/embedded sys midterm report/results/accuracy/train_logs"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
mkdir -p "$L/exp15"; cp "$S/exp15"/ft30_cand_*.json "$L/exp15/"
cd "D:/115-1/embedded sys midterm report/code" || exit 1
for m in 0 0.1 0.2; do
  PTS=""; for c in 125 111.11 100 90.91 83.33 76.92 71.43 66.67 50; do PTS="$PTS $c=$C/points_tsr_v2c_m${m}_f$c.npz"; done
  "$S/venv_finn010/Scripts/python.exe" timing_safe.py table --acc "$C/acc_model_3rep.json" \
      --sweep "$C/threshold_sweep_tsr_v2c_m${m}_merged.json" --points $PTS --logs "$L"/exp* \
      --extra 44444444=83.94,83.83 --out "$C/main_table_tsr_v2c_m$m.json" > "$C/main_table_tsr_v2c_m$m.log" 2>&1
  echo "== routing margin $m"; grep -A6 "margin 5:" "$C/main_table_tsr_v2c_m$m.log" | head -7
done
