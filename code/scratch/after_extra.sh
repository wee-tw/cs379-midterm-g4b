#!/bin/bash
# Follow-ups added on 10/30:
#  1) idle-policy analysis under the accuracy constraint (numpy only, runs as soon as the
#     accuracy model exists),
#  2) robustness of the search to budget derating and clock set,
#  3) CNV-W1A1 estimate-vs-implementation check against MLPerf Tiny v0.7 (imports torch).
# 2) and 3) wait until the overnight trainings are done, to stay inside the paging file.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== constrained search" "$S/after_exp3.log" 2>/dev/null; do sleep 30; done
echo "== idle policies with accuracy $(date)"
"$PY" -u energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model.json" \
    --out "$S/cost/idle_acc.json" > "$S/cost/idle_acc.log" 2>&1
"$PY" -u idle_summary.py "$S/cost/idle_acc.json" >> "$S/cost/idle_acc.log" 2>&1
tail -12 "$S/cost/idle_acc.log"
until grep -q "== done" "$S/after_exp3.log" 2>/dev/null; do sleep 60; done
echo "== robustness $(date)"
bash "$S/robustness.sh" > "$S/cost/robustness.log" 2>&1
tail -2 "$S/cost/robustness.log"
echo "== CNV calibration $(date)"
"$PY" -u calib_cnv_mlperf.py --out "$S/cnv_calib" > "$S/cnv_calib.log" 2>&1
tail -30 "$S/cnv_calib.log"
echo "== done $(date)"
