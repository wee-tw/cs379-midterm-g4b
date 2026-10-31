#!/bin/bash
# Threshold-consistent design spaces (threshold_consistent.py on tables_corr_{lo,hi}).
V="$1"
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$CODE"
T="$S/cost/tables_thr_$V.json"; P="$S/cost/points_thr_$V.npz"; L="$S/cost/latency_thr_$V.npz"
"$PY" energy_search.py points --tables "$T" --out "$P" > "$S/cost/points_thr_$V.log" 2>&1
tail -1 "$S/cost/points_thr_$V.log"
"$PY" latency_check.py points --tables "$T" --points "$P" --out "$L"
echo "== [$V] main search (3-replicate model, margin 1)"
"$PY" energy_search.py search --points "$P" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/search_thr_$V.json" > "$S/cost/search_thr_$V.log" 2>&1
grep -A6 "power params: nominal" "$S/cost/search_thr_$V.log" | head -7 | cut -c1-200
echo "== [$V] threshold sweep with latency-aware policies"
"$PY" threshold_sweep.py --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --out "$S/cost/threshold_sweep_thr_$V.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-200
echo "== [$V] latency-aware idle analysis"
"$PY" energy_search.py idle --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --lat-scale 0.6024 --out "$S/cost/idle_lat_thr_$V.json" > "$S/cost/idle_lat_thr_$V.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_thr_$V.json" | grep -E "^== nominal|^== z2|R=  300|R= 3000" | cut -c1-240
echo "== [$V] proxies"
"$PY" proxy_finn.py --points "$P" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/proxy_finn_thr_$V.json" | grep -E "margin|tau\(BOPs |tau\(weight bits " | cut -c1-260
echo "== [$V] key designs"
"$PY" "$S/compare_designs.py" "$P" 44444448 44444488 44242448 84242448 44224448 42442448 48484848
echo "== [$V] done $(date +%T)"
