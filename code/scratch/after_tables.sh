#!/bin/bash
# Runs once cost/tables.json exists: validation, design points, accuracy-free search,
# proxy analysis, hardware figures.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$CODE"
until [ -f "$S/cost/tables.json" ]; do sleep 20; done
echo "== validate $(date)"
"$PY" -u assemble_costs.py validate --tables "$S/cost/tables.json" --n 8 > "$S/cost/validate.log" 2>&1
tail -3 "$S/cost/validate.log"
echo "== points $(date)"
"$PY" -u energy_search.py points --tables "$S/cost/tables.json" --out "$S/cost/points.npz" > "$S/cost/points.log" 2>&1
tail -2 "$S/cost/points.log"
echo "== search without accuracy constraint $(date)"
"$PY" -u energy_search.py search --points "$S/cost/points.npz" --out "$S/cost/search_noacc.json" > "$S/cost/search_noacc.log" 2>&1
tail -40 "$S/cost/search_noacc.log"
echo "== proxy analysis $(date)"
"$PY" -u analysis_proxy.py --points "$S/cost/points.npz" --out "$S/cost/proxy_noacc.json" > "$S/cost/proxy_noacc.log" 2>&1
echo "== figures $(date)"
"$PY" -u make_figures.py --points "$S/cost/points.npz" --recipe "$S/exp2" --out "$CODE/results/figs" > "$S/cost/figs.log" 2>&1
tail -2 "$S/cost/figs.log"
echo "== done $(date)"
