#!/bin/bash
# exp7: third replicate (seed 2) of the sensitivity scan, margin-3 candidate seeds, 60-epoch
# uniform W4A8; then the accuracy model over 3 replicates and the analyses repeated.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_exp6b.log" 2>/dev/null; do sleep 60; done
echo "== exp7 $(date)"
"$PY" run_queue.py "$S/exp7_jobs.json" --out "$S/exp7" --parallel 2 > "$S/exp7_queue.log" 2>&1
tail -5 "$S/exp7_queue.log"
echo "== accuracy model over 3 replicates $(date)"
"$PY" acc_model.py --runs "$S/exp3" "$S/exp6" "$S/exp7" --base sens_base --reference uniform_model \
    --margin 1.0 --check "val_*" --out "$S/cost/acc_model_3rep.json" | tee "$S/cost/acc_model_3rep.log"
"$PY" energy_search.py search --points "$S/cost/points.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/search_acc_3rep.json" > "$S/cost/search_acc_3rep.log" 2>&1
"$PY" energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/idle_acc_3rep.json" > "$S/cost/idle_acc_3rep.log" 2>&1
"$PY" idle_summary.py "$S/cost/idle_acc_3rep.json" >> "$S/cost/idle_acc_3rep.log" 2>&1
"$PY" threshold_sweep.py --points "$S/cost/points.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/threshold_sweep_3rep.json" > "$S/cost/threshold_sweep_3rep.log" 2>&1
grep -E "^== margin|cont vs" "$S/cost/threshold_sweep_3rep.log" | cut -c1-170
grep "cont gap" "$S/cost/idle_acc_3rep.log"
echo "== done $(date)"
