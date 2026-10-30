#!/bin/bash
# DSE-harm study, remedies and mechanism (12.25-12.27, 12.32) with the second LUT calibration as truth:
# every view gets calib_v2's HLS-MVU, SWG and small-operator LUTs; the RTL-MVU lane-count model goes
# to the views that model the RTL MVU's LUTs (reference = truth, lut_fixed, bram_gap); FINN's views
# (finn, dsp_fixed, finn_all) keep FINN's 0. Two point runs at a time (Vivado jobs share the memory).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
C="$S/cost"; FIT="$C/calib_v2.json"; A="$C/acc_model_3rep.json"
echo "== tables $(date +%T)"
for v in reference lut_fixed bram_gap; do
  "$PY" calib_v2.py tables --override --fit "$FIT" --tables "$C/tables_${v}_cal.json" --scratch "$S" --out "$C/tables_${v}_calv2.json" | cut -c1-120
done
for v in finn dsp_fixed finn_all; do
  "$PY" calib_v2.py tables --override --fit "$FIT" --keep-rtl --tables "$C/tables_${v}_cal.json" --scratch "$S" --out "$C/tables_${v}_calv2.json" | cut -c1-120
done
run() { "$PY" energy_search.py points --tables "$C/tables_$1_calv2.json" --out "$C/points_$1_calv2.npz" > "$C/points_$1_calv2.log" 2>&1; echo "$1: $(tail -1 "$C/points_$1_calv2.log") $(date +%T)"; }
run reference & run finn & wait
run lut_fixed & run dsp_fixed & wait
run bram_gap & run finn_all & wait
echo "== harm $(date +%T)"
"$PY" dse_harm.py --truth "$C/points_reference_calv2.npz" \
  --biased finn="$C/points_finn_calv2.npz" lut_fixed="$C/points_lut_fixed_calv2.npz" dsp_fixed="$C/points_dsp_fixed_calv2.npz" \
    bram_gap="$C/points_bram_gap_calv2.npz" finn_all="$C/points_finn_all_calv2.npz" \
  --acc "$A" --out "$C/dse_harm_calv2.json" > "$C/dse_harm_calv2.log" 2>&1
tail -60 "$C/dse_harm_calv2.log"
echo "== remedies $(date +%T)"
"$PY" dse_remedies.py --truth "$C/points_reference_calv2.npz" --biased "$C/points_finn_calv2.npz" --acc "$A" \
  --out "$C/dse_remedies_calv2.json" > "$C/dse_remedies_calv2.log" 2>&1
cat "$C/dse_remedies_calv2.log" | cut -c1-220
echo "== done $(date +%T)"
