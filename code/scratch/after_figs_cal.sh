#!/bin/bash
# Paper figures on the calibrated costs (tables_calov, implementation_notes.md 12.31). The
# figures they replace are kept in results/figs/pre_calibration/.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
mkdir -p results/figs/pre_calibration
for f in threshold_sweep_nominal lut_vs_throughput energy_vs_accuracy_nominal energy_vs_accuracy_z2_anchor \
         energy_vs_accuracy_flat energy_vs_accuracy_bismo_fit latency_policies_nominal latency_policies_z2_anchor \
         h3_pairs_nominal h3_pairs_z2_anchor; do
  [ -f "results/figs/$f.png" ] && cp "results/figs/$f.png" "results/figs/pre_calibration/$f.png"
done
P="$S/cost/points_calov.npz"; L="$S/cost/latency_calov.npz"
echo "== latency-aware idle analysis $(date +%T)"
"$PY" energy_search.py idle --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --lat-scale 0.6024 --out "$S/cost/idle_lat_calov.json" > "$S/cost/idle_lat_calov.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_calov.json" | grep -E "^== nominal|^== z2|R=  300|R= 3000" | cut -c1-240
echo "== figures $(date +%T)"
"$PY" make_figures.py --points "$P" --acc "$S/cost/acc_model_3rep.json" --sweep "$S/cost/threshold_sweep_calov.json" \
    --latency "$S/cost/idle_lat_calov.json" --protocol "$S/cost/cand_table_3rep.json" --out results/figs 2>&1 | tail -n 2 | cut -c1-300
echo "== done $(date +%T)"
