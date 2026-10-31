#!/bin/bash
# FINN numerics check (Brevitas vs QONNX vs streamlined vs HW layers) for the final A' checkpoints.
# usage: numerics_final.sh <n images> <tag>
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
PY="$S/venv_finn010/Scripts/python.exe"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
N=$1; TAG=$2
cd "$CODE" || exit 1
run() {  # run <bits string> <8 bits>
  echo "== $1 n=$N start $(date +%T)"
  "$PY" check_finn_numerics.py --bits ${@:2} --ckpt "$S/exp17/ft30_cand_$1.pt" --npz "$S/cifar_hf/cifar10.npz" \
      --recipe lowbit --n "$N" 2>&1 | grep -v "^ *warn\|Warning" 
  echo "== $1 n=$N end $(date +%T) rc=${PIPESTATUS[0]}"
}
run 44244448 4 4 2 4 4 4 4 8 > "$S/numerics_final_${TAG}_44244448.log" 2>&1 &
run 22244448 2 2 2 4 4 4 4 8 > "$S/numerics_final_${TAG}_22244448.log" 2>&1 &
wait
echo "== numerics $TAG all done $(date +%T)"
