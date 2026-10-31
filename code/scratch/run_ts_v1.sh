#!/bin/bash
# Timing-safe main results, filter version (notes 12.35; every step tested on 400 configs in
# $S/ts_test). ts100: no folded MVAU_hls wider than SIMD 27 (clocks <= 100 MHz); ts125: no
# folded MVAU_hls at all (clocks <= 125 MHz). Stops at the first failed step.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
C="$S/cost"; A="$C/acc_model_3rep.json"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
cd "$CODE" || exit 1
step() { echo "== $1 $(date +%T)"; }
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }

step tables
"$PY" timing_safe.py tables --tables "$C/tables_calv2.json" --prefix "$C/tables_calv2" || exit 1
need "$C/tables_calv2_ts100.json"; need "$C/tables_calv2_ts125.json"

step "points (both in parallel)"
for v in ts100 ts125; do
  "$PY" energy_search.py points --tables "$C/tables_calv2_$v.json" --out "$C/points_calv2_$v.npz" \
      > "$C/points_calv2_$v.log" 2>&1 &
done
wait
for v in ts100 ts125; do need "$C/points_calv2_$v.npz"; tail -1 "$C/points_calv2_$v.log"; done

step latency
for v in ts100 ts125; do
  "$PY" latency_check.py points --tables "$C/tables_calv2_$v.json" --points "$C/points_calv2_$v.npz" \
      --out "$C/latency_calv2_$v.npz" | tail -1
  need "$C/latency_calv2_$v.npz"
done

step "merged sweep (ts100 <= 100 MHz, ts125 <= 125 MHz)"
"$PY" timing_safe.py sweep --p100 "$C/points_calv2_ts100.npz" --p125 "$C/points_calv2_ts125.npz" \
    --acc "$A" --out "$C/threshold_sweep_ts_merged.json" || exit 1

step "threshold sweep with idle policies, each space at its own cap"
"$PY" threshold_sweep.py --points "$C/points_calv2_ts100.npz" --acc "$A" --latency "$C/latency_calv2_ts100.npz" \
    --fmax 100 --out "$C/threshold_sweep_ts100.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-220
"$PY" threshold_sweep.py --points "$C/points_calv2_ts125.npz" --acc "$A" --latency "$C/latency_calv2_ts125.npz" \
    --fmax 125 --out "$C/threshold_sweep_ts125.json" | grep -E "^== margin|cont vs" | cut -c1-220
echo "== done $(date +%T)"
