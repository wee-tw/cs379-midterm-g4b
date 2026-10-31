#!/bin/bash
# exp18: seeds 3 and 4 for the two borderline configs (44444444 = B1 at margin 1; 22244448 = A' at margins 5-8),
# then the main tables with 5 seeds for them (written next to the official ones, not over them).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
C="$S/cost"; L="D:/115-1/embedded sys midterm report/results/accuracy/train_logs"; PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
cd "$CODE" || exit 1
echo "== exp18 queue $(date +%T)"
"$PY" run_queue.py "$S/exp18_jobs.json" --out "$S/exp18" --parallel 1 > "$S/exp18_queue.log" 2>&1
n=$(grep -c "^[0-9:]* finished ft30_cand.* exit 0" "$S/exp18_queue.log")
echo "== exp18 trainings ok: $n / 4 $(date +%T)"
[ "$n" -eq 4 ] || { echo "exp18 did not complete"; tail -5 "$S/exp18_queue.log"; exit 1; }
mkdir -p "$L/exp18" && cp "$S/exp18"/ft30_cand_*.json "$L/exp18/"
for tag in tsr_v1218 tsr_v1218_m0.1; do
  out=$([ "$tag" = tsr_v1218 ] && echo m0 || echo m0.1)
  PTS=""; for c in 125 111.11 100 90.91 83.33 76.92 71.43 66.67 50; do PTS="$PTS $c=$C/points_${tag}_f$c.npz"; done
  "$PY" timing_safe.py table --acc "$C/acc_model_3rep.json" --sweep "$C/threshold_sweep_${tag}_merged.json" \
      --points $PTS --logs "$L"/exp* --out "$S/main_table_${tag}_5seed.json" > "$S/main_table_${tag}_5seed.log" 2>&1
  echo "== table $out rc=$? $(date +%T)"
done
echo "== exp18 all done $(date +%T)"
