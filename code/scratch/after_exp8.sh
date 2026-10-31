#!/bin/bash
# exp8: progressive fine-tunes (W4A4 -> 30 epochs, KD, lr 0.02) of the margin-2/3 picks of
# the 2-replicate accuracy model and their BOPs proxies, plus A' and uniform W4A8 under the
# same protocol, so H3 can be compared within one training protocol.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_exp7.log" 2>/dev/null; do sleep 60; done
echo "== exp8 $(date)"
"$PY" run_queue.py "$S/exp8_jobs.json" --out "$S/exp8" --parallel 2 > "$S/exp8_queue.log" 2>&1
tail -12 "$S/exp8_queue.log"
echo "== done $(date)"
