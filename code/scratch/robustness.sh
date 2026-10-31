#!/bin/bash
# Robustness of the search conclusions to the budget derating and the clock set.
# Idempotent: steps whose output already exists are skipped (after_extra.sh calls it again).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
cd "D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
for d in "0.9 0.8" "0.7 0.6"; do
  set -- $d
  tag="lut$1_bram$2"
  [ -f "$S/cost/points_$tag.npz" ] || "$PY" energy_search.py points --tables "$S/cost/tables.json" \
      --derate-lut $1 --derate-bram $2 --out "$S/cost/points_$tag.npz" > "$S/cost/points_$tag.log" 2>&1
  [ -f "$S/cost/search_$tag.json" ] || "$PY" energy_search.py search --points "$S/cost/points_$tag.npz" \
      --acc "$S/cost/acc_model.json" --out "$S/cost/search_$tag.json" > "$S/cost/search_$tag.log" 2>&1
  [ -f "$S/cost/idle_$tag.json" ] || { "$PY" energy_search.py idle --points "$S/cost/points_$tag.npz" \
      --acc "$S/cost/acc_model.json" --out "$S/cost/idle_$tag.json" > "$S/cost/idle_$tag.log" 2>&1;
      "$PY" idle_summary.py "$S/cost/idle_$tag.json" >> "$S/cost/idle_$tag.log" 2>&1; }
done
[ -f "$S/cost/search_f143.json" ] || "$PY" energy_search.py search --points "$S/cost/points.npz" \
    --acc "$S/cost/acc_model.json" --fset 50 66.67 100 125 142.86 \
    --out "$S/cost/search_f143.json" > "$S/cost/search_f143.log" 2>&1
echo done
