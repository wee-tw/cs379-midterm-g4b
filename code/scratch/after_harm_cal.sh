#!/bin/bash
# DSE-harm study (12.25, 12.27) on the calibrated costs (12.31): truth = reference with the
# Vivado-calibrated RTL-MVU LUTs; all models get the HLS calibration of MVAU_hls; the replay-
# buffer BRAM goes to every model except the two that leave it out on purpose (bram_gap = the
# reference without it, finn_all = FINN's view with all three gaps).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
until grep -q "== done" "$S/after_cal2.log" 2>/dev/null; do sleep 60; done
echo "== calibrated harm tables $(date +%T)"
C="$S/cost"; HLS="--audit $S/audit_hls_finn $S/audit_hls"
for v in finn dsp_fixed lut_fixed reference; do
  src="$C/tables_${v}_thr.json"
  if [ "$v" = lut_fixed ] || [ "$v" = reference ]; then
    "$PY" calib_mvu_vivado.py tables --audit "$S/audit" --tables "$src" --out "$C/h_${v}_1.json" | tail -1
    src="$C/h_${v}_1.json"
  fi
  "$PY" calib_mvau_hls.py tables $HLS --tables "$src" --out "$C/h_${v}_2.json" | tail -1
  "$PY" replay_bram.py tables --tables "$C/h_${v}_2.json" --out "$C/tables_${v}_cal.json" | tail -1
done
cp "$C/h_reference_2.json" "$C/tables_bram_gap_cal.json"; cp "$C/h_finn_2.json" "$C/tables_finn_all_cal.json"
rm -f "$C"/h_*_1.json
run() { "$PY" energy_search.py points --tables "$C/tables_$1_cal.json" --out "$C/points_$1_cal.npz" > "$C/points_$1_cal.log" 2>&1; echo "$1: $(tail -1 "$C/points_$1_cal.log")"; }
run reference & run finn & run lut_fixed & wait
run dsp_fixed & run bram_gap & run finn_all & wait
rm -f "$C"/h_*_2.json
"$PY" dse_harm.py --truth "$C/points_reference_cal.npz" \
  --biased finn="$C/points_finn_cal.npz" lut_fixed="$C/points_lut_fixed_cal.npz" dsp_fixed="$C/points_dsp_fixed_cal.npz" \
    bram_gap="$C/points_bram_gap_cal.npz" finn_all="$C/points_finn_all_cal.npz" \
  --acc "$C/acc_model_3rep.json" --out "$C/dse_harm_cal.json" > "$C/dse_harm_cal.log" 2>&1
tail -60 "$C/dse_harm_cal.log"
echo "== done $(date +%T)"
