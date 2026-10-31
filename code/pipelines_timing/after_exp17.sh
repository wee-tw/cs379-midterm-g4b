#!/bin/bash
# After exp17 (44244448, 22244448; 3 seeds each): copy the logs, rebuild the main tables of the
# routing-checked refold with ft30 logs only, then synthesize the final A' / B1 designs (vd6)
# and the B2 designs (vd7) node by node.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
C="$S/cost"; L="D:/115-1/embedded sys midterm report/results/accuracy/train_logs"; PY="$S/venv_finn010/Scripts/python.exe"
CLASSES="125 111.11 100 90.91 83.33 76.92 71.43 66.67 50"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
until [ "$(grep -c '^[0-9:]* finished ft30_cand' "$S/exp17_queue.log" 2>/dev/null)" -ge 6 ]; do sleep 30; done
mkdir -p "$L/exp17"; cp "$S/exp17"/ft30_cand_*.json "$L/exp17/"
echo "== exp17 $(date +%T)"; grep "^ft30_cand" "$S/exp17_queue.log"
until grep -q "^== all done" "$S/after_v5.log" 2>/dev/null; do sleep 30; done
cd "D:/115-1/embedded sys midterm report/code" || exit 1
for tag in tsr_v1218 tsr_v1218_m0.1; do
  out=$([ "$tag" = tsr_v1218 ] && echo tsr_v1218_m0 || echo "$tag")
  PTS=""; for c in $CLASSES; do PTS="$PTS $c=$C/points_${tag}_f$c.npz"; done
  "$PY" timing_safe.py table --acc "$C/acc_model_3rep.json" --sweep "$C/threshold_sweep_${tag}_merged.json" \
      --points $PTS --logs "$L"/exp* --out "$C/main_table_$out.json" > "$C/main_table_$out.log" 2>&1
  echo "== main table $out $(date +%T)"; grep -E "margin|B1_measured|B2_measured" "$C/main_table_$out.log" | head -26
done
rm -rf D:/vd6 D:/vw6
bash "$S/after_v5_verify.sh" > "$S/after_v5_verify.log" 2>&1 &
bash "$S/after_vd7.sh" > "$S/after_vd7.log" 2>&1 &
wait
echo "== exp17 follow-up done $(date +%T)"
