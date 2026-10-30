#!/bin/bash
# C2 robustness to the truth's LUT error (12.34): the v2.1 reference with every LUT x0.90 and x1.10,
# against FINN's views (gap-isolating finn_calv2 and raw FINN). Waits for exp12 (memory).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
C="$S/cost"; A="$C/acc_model_3rep.json"
until grep -q "finished ft30_cand_84848484_s2" "$S/exp12_queue.log" 2>/dev/null; do sleep 30; done
for s in 090 110; do
  "$PY" calib_sensitivity.py tables --tables "$C/tables_reference_calv2.json" --all $( [ "$s" = 090 ] && echo 0.90 || echo 1.10 ) --out "$C/tables_ref_lut$s.json" | tail -1
  "$PY" energy_search.py points --tables "$C/tables_ref_lut$s.json" --out "$C/points_ref_lut$s.npz" > "$C/points_ref_lut$s.log" 2>&1
  rm -f "$C/tables_ref_lut$s.json"
  echo "== truth LUT x$s $(date +%T)"
  "$PY" dse_harm.py --truth "$C/points_ref_lut$s.npz" --biased finn="$C/points_finn_calv2.npz" finn_raw="$C/points_finn.npz" \
      --acc "$A" --power nominal --out "$C/dse_harm_ref_lut$s.json" 2>&1 | grep -E "biased model|margin [0-9]:" | cut -c1-200
  "$PY" feasible_at_k.py --truth "$C/points_ref_lut$s.npz" --biased finn="$C/points_finn_calv2.npz" finn_raw="$C/points_finn.npz" --acc "$A" | tail -10
done
echo "== done $(date +%T)"
