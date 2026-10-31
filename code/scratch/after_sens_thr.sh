#!/bin/bash
# 12.23/12.25/12.26 analyses rerun on the threshold-consistent costs (12.27), so the paper's
# numbers all come from one cost model. Derated-budget runs (mitigation, ZU3EG) are not redone.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
A="$S/cost/acc_model_3rep.json"
for V in lo hi; do for M in 1 2; do
  echo "== dsp weight sensitivity $V margin $M"
  "$PY" dsp_weight_sens.py --points "$S/cost/points_thr_$V.npz" --acc "$A" --margin $M \
      --out "$S/cost/dsp_weight_sens_thr_${V}_m$M.json" 2>&1 | tail -8 | cut -c1-220
done; done
echo "== families"
"$PY" dse_families.py --truth "$S/cost/points_reference_thr.npz" --biased "$S/cost/points_finn_thr.npz" \
    --acc "$A" --out "$S/cost/dse_families_thr.json" 2>&1 | tail -30 | cut -c1-240
echo "== controls (misjudged rate and tau of the consistent FINN model)"
"$PY" dse_control.py --truth "$S/cost/points_reference_thr.npz" --acc "$A" --rate 0.0122 --tau 0.9717 \
    --out "$S/cost/dse_control_thr.json" 2>&1 | tail -20 | cut -c1-240
echo "== done $(date +%T)"
