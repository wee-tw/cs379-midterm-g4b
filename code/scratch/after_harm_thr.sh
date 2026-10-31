#!/bin/bash
# DSE-harm study (12.25) rerun on threshold-consistent costs (threshold_consistent.py applied to
# all four cost models alike), to check that 4f does not depend on the free threshold costs.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
for v in finn dsp_fixed lut_fixed reference; do
  ( "$PY" threshold_consistent.py --tables "$S/cost/tables_$v.json" --out "$S/cost/tables_${v}_thr.json" &&
    "$PY" energy_search.py points --tables "$S/cost/tables_${v}_thr.json" --out "$S/cost/points_${v}_thr.npz" \
      > "$S/cost/points_${v}_thr.log" 2>&1 && echo "$v points: $(tail -1 "$S/cost/points_${v}_thr.log")" ) &
done
wait
"$PY" dse_harm.py --truth "$S/cost/points_reference_thr.npz" \
  --biased finn="$S/cost/points_finn_thr.npz" lut_fixed="$S/cost/points_lut_fixed_thr.npz" dsp_fixed="$S/cost/points_dsp_fixed_thr.npz" \
  --acc "$S/cost/acc_model_3rep.json" --out "$S/cost/dse_harm_thr.json" > "$S/cost/dse_harm_thr.log" 2>&1
tail -40 "$S/cost/dse_harm_thr.log"
echo "== done $(date +%T)"
