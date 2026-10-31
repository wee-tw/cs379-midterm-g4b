#!/bin/bash
# After after_exp7.sh: candidate table and protocol figure with the 3-replicate model,
# the latency-aware idle analysis and threshold sweep with it, and the key numbers.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$CODE"
echo "== after_exp7 log (tail)"
tail -25 "$S/after_exp7.log" | cut -c1-200
echo "== exp7 named runs"
for f in "$S"/exp7/*.json "$S"/exp8/*.json; do [ -f "$f" ] || continue
  n=$(basename "$f" .json); case "$n" in sens_*) continue;; esac
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d['final_acc'])" "$f" "$n"
done
echo "== candidate table (1/2/3 replicates)"
"$PY" cand_table.py --acc "1rep=$S/cost/acc_model.json" "2rep=$S/cost/acc_model_2rep.json" \
    "3rep=$S/cost/acc_model_3rep.json" --dirs "$S/exp3" "$S/exp4" "$S/exp5" "$S/exp6" "$S/exp7" \
    $([ -d "$S/exp8" ] && echo "$S/exp8") \
    --out "$S/cost/cand_table_3rep.md" --json "$S/cost/cand_table_3rep.json" | tail -40
"$PY" make_figures.py --protocol "$S/cost/cand_table_3rep.json" --protocol-model 3rep \
    --out "$CODE/results/figs" | tail -1
"$PY" -c "
import sys, numpy as np, make_figures as mf
for pname in ('nominal', 'z2_anchor'):
    mf.fig_h3_pairs(np.load(sys.argv[1])['rows'], sys.argv[2], [
        ('margin 2, 1-rep model', '88242488', '44242888'),
        ('margin 3, 1-rep model', '84242444', '44242428'),
        ('margin 2, 2/3-rep model', '84242448', '44242448'),
        ('margin 3, 2-rep model', '84242248', '42442488')], sys.argv[3], pname)
print('h3_pairs figures written')
" "$S/cost/points.npz" "$S/cost/cand_table_3rep.json" "$CODE/results/figs"
echo "== latency-aware idle and threshold sweep, 3-replicate model"
"$PY" energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model_3rep.json" \
    --latency "$S/cost/latency.npz" --lat-scale 0.6024 --out "$S/cost/idle_lat_3rep.json" \
    > "$S/cost/idle_lat_3rep.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_3rep.json" | cut -c1-260
"$PY" threshold_sweep.py --points "$S/cost/points.npz" --acc "$S/cost/acc_model_3rep.json" \
    --latency "$S/cost/latency.npz" --out "$S/cost/threshold_sweep_lat_3rep.json" \
    | grep -E "^== margin|cont vs|jit:" | cut -c1-200
echo "== done $(date +%T)"
