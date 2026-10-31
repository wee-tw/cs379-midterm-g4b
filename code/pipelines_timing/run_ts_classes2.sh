#!/bin/bash
# Timing classes on an already calibrated table (no refold), e.g. tables_calv2.json:
#   run_ts_classes2.sh <calibrated tables> <tag> <rtl timing json> <parallel points jobs> <margin> [tables options ...]
# As run_ts_classes.sh; the extra arguments go to timing_safe.py tables, e.g.
#   --op-margin MVAU_rtl=0.2 MVAU_hls=0.02   or   --hls-routed-simd 18
# Same steps as the second half of run_ts_v3.sh (tested there).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
C="$S/cost"; A="$C/acc_model_3rep.json"; T="$1"; TAG="$2"; RTL="$3"; JOBS="${4:-3}"; MARGIN="${5:-0}"; shift 5
CLASSES="125 111.11 100 90.91 83.33 76.92 71.43 66.67 50"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
cd "$CODE" || exit 1
step() { echo "== $1 $(date +%T)"; }
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
need "$T"; need "$RTL"
step "timing classes"
"$PY" timing_safe.py tables --tables "$T" --rtl-timing "$RTL" --margin "$MARGIN" "$@" --prefix "$C/tables_$TAG" || exit 1
step "points ($JOBS at a time)"
n=0
for c in $CLASSES; do
  need "$C/tables_${TAG}_f$c.json"
  "$PY" energy_search.py points --tables "$C/tables_${TAG}_f$c.json" --out "$C/points_${TAG}_f$c.npz" \
      > "$C/points_${TAG}_f$c.log" 2>&1 &
  n=$((n + 1)); [ $((n % JOBS)) -eq 0 ] && wait
done
wait
for c in $CLASSES; do need "$C/points_${TAG}_f$c.npz"; echo "f$c: $(tail -1 "$C/points_${TAG}_f$c.log" | grep -o '[0-9]* points')"; done
step "merged sweep"
PTS=""; for c in $CLASSES; do PTS="$PTS $c=$C/points_${TAG}_f$c.npz"; done
"$PY" timing_safe.py sweep --acc "$A" --out "$C/threshold_sweep_${TAG}_merged.json" --points $PTS || exit 1
echo "== done $(date +%T)"
