#!/bin/bash
# After after_exp6.sh: candidate table with the 1- and 2-replicate accuracy models, the
# latency-aware idle analysis with the 2-replicate model, and the key numbers.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$CODE"
echo "== after_exp6 log (tail)"
tail -25 "$S/after_exp6.log" | cut -c1-200
echo "== candidate table"
"$PY" cand_table.py --acc "1rep=$S/cost/acc_model.json" "2rep=$S/cost/acc_model_2rep.json" \
    --dirs "$S/exp3" "$S/exp4" "$S/exp5" "$S/exp6" --out "$S/cost/cand_table_2rep.md" \
    --json "$S/cost/cand_table_2rep.json" | tail -22
echo "== latency-aware idle, 2-replicate model"
"$PY" energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model_2rep.json" \
    --latency "$S/cost/latency.npz" --lat-scale 0.6024 --out "$S/cost/idle_lat_2rep.json" \
    > "$S/cost/idle_lat_2rep.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_2rep.json" | cut -c1-260
echo "== done $(date +%T)"
