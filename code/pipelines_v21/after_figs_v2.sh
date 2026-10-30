#!/bin/bash
# Paper figures on the second LUT calibration (tables_calv2, notes 12.33). The v1-calibration figures
# they replace go to results/figs/v1_calibration/.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MPLBACKEND=Agg
cd "$CODE"
C="$S/cost"; A="$C/acc_model_3rep.json"
mkdir -p results/figs/v1_calibration
for f in threshold_sweep_nominal lut_vs_throughput energy_vs_accuracy_nominal energy_vs_accuracy_z2_anchor \
         energy_vs_accuracy_flat energy_vs_accuracy_bismo_fit latency_policies_nominal latency_policies_z2_anchor \
         h3_pairs_nominal budget_cliff_nominal estimator_error_concentration_nominal; do
  [ -f "results/figs/$f.png" ] && cp -n "results/figs/$f.png" "results/figs/v1_calibration/$f.png"
done
"$PY" make_figures.py --points "$C/points_calv2.npz" --acc "$A" --sweep "$C/threshold_sweep_calv2.json" \
    --latency "$C/idle_lat_calv2.json" --protocol "$C/cand_table_3rep.json" --budget "$C/budget_sweep_calv2.json" \
    --out results/figs 2>&1 | tail -n 1 | cut -c1-200
"$PY" -c "
import sys, make_figures as mf
mf.fig_concentration(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
print('concentration figure written')
" "$C/points_finn_calv2.npz" "$C/points_reference_calv2.npz" "$A" results/figs
echo "== done $(date +%T)"
