#!/bin/bash
# exp6: seed-1 replicate of the full sensitivity scan + candidate seeds; then the accuracy
# model averaged over the two replicates and the analyses repeated with it.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_exp5.log" 2>/dev/null; do sleep 60; done
echo "== exp6 $(date)"
"$PY" run_queue.py "$S/exp6_jobs.json" --out "$S/exp6" --parallel 2 > "$S/exp6_queue.log" 2>&1
tail -5 "$S/exp6_queue.log"
echo "== accuracy model over 2 replicates $(date)"
"$PY" acc_model.py --runs "$S/exp3" "$S/exp6" --base sens_base --reference uniform_model \
    --margin 1.0 --check "val_*" --out "$S/cost/acc_model_2rep.json" | tee "$S/cost/acc_model_2rep.log"
"$PY" -I -c "
import json; a=json.load(open(r'$S/cost/acc_model_2rep.json'))
sp=sorted(a['delta_spread'].items(), key=lambda kv: -kv[1])
print('replicates', a['replicates'], 'largest seed spread of deltas:', sp[:6])
print('median spread', sorted(v for _, v in sp)[len(sp)//2])"
"$PY" energy_search.py search --points "$S/cost/points.npz" --acc "$S/cost/acc_model_2rep.json" \
    --out "$S/cost/search_acc_2rep.json" > "$S/cost/search_acc_2rep.log" 2>&1
"$PY" energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model_2rep.json" \
    --out "$S/cost/idle_acc_2rep.json" > "$S/cost/idle_acc_2rep.log" 2>&1
"$PY" idle_summary.py "$S/cost/idle_acc_2rep.json" >> "$S/cost/idle_acc_2rep.log" 2>&1
"$PY" threshold_sweep.py --points "$S/cost/points.npz" --acc "$S/cost/acc_model_2rep.json" \
    --out "$S/cost/threshold_sweep_2rep.json" > "$S/cost/threshold_sweep_2rep.log" 2>&1
grep -E "^== margin|cont vs" "$S/cost/threshold_sweep_2rep.log" | cut -c1-170
grep "cont gap" "$S/cost/idle_acc_2rep.log"
echo "== done $(date)"
