#!/bin/bash
# Timing-safe main results with refolded MVAU_hls (notes 12.35). Usage:
#   run_ts_v2.sh <raw tables from cost_tables.py --ts-refold (lut, dspc, dsp4c)> <tag> [limit]
# lut_finn comes unchanged from tables.json (its SIMD is already <= 18). The raw tables go
# through the validated calibration chain (calib_chain.sh reproduces tables_calv2.json exactly),
# then timing_safe.py splits them into ts100 / ts125 and the searches run as in run_ts_v1.sh.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
C="$S/cost"; A="$C/acc_model_3rep.json"; RAW="$1"; TAG="tsr_$2"; LIM="${3:+--limit $3}"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
cd "$CODE" || exit 1
step() { echo "== $1 $(date +%T)"; }
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
need "$RAW"

step "lut source: refolded lut + original lut_finn"
"$PY" - "$RAW" "$C/tables.json" "$C/tables_lutsrc_$TAG.json" <<'PYEOF'
import json, sys
r, o = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
t = {k: v for k, v in r["table"].items() if k.startswith("lut|")}
t.update({k: v for k, v in o["table"].items() if k.startswith("lut_finn|")})
json.dump({"T_GRID": o["T_GRID"], "modes": ["lut", "lut_finn"], "table": t,
           "mismatches": r["mismatches"]}, open(sys.argv[3], "w"))
print("lut", sum(k.startswith("lut|") for k in t), "lut_finn", sum(k.startswith("lut_finn|") for k in t))
PYEOF
need "$C/tables_lutsrc_$TAG.json"

step "calibration chain"
bash "$S/calib_chain.sh" "$C/tables_lutsrc_$TAG.json" "$RAW" "$TAG" | grep -E "^==|records|MISSING" | cut -c1-160
need "$C/tables_calv2_$TAG.json"

step "timing classes"
"$PY" timing_safe.py tables --tables "$C/tables_calv2_$TAG.json" --prefix "$C/tables_calv2_$TAG" || exit 1
for v in ts100 ts125; do need "$C/tables_calv2_${TAG}_$v.json"; done

step "points (both in parallel)"
for v in ts100 ts125; do
  "$PY" energy_search.py points --tables "$C/tables_calv2_${TAG}_$v.json" $LIM \
      --out "$C/points_${TAG}_$v.npz" > "$C/points_${TAG}_$v.log" 2>&1 &
done
wait
for v in ts100 ts125; do need "$C/points_${TAG}_$v.npz"; tail -1 "$C/points_${TAG}_$v.log" | cut -c1-160; done

step latency
for v in ts100 ts125; do
  "$PY" latency_check.py points --tables "$C/tables_calv2_${TAG}_$v.json" --points "$C/points_${TAG}_$v.npz" \
      --out "$C/latency_${TAG}_$v.npz" | tail -1
  need "$C/latency_${TAG}_$v.npz"
done

step "merged sweep (ts100 <= 100 MHz, ts125 <= 125 MHz)"
"$PY" timing_safe.py sweep --p100 "$C/points_${TAG}_ts100.npz" --p125 "$C/points_${TAG}_ts125.npz" \
    --acc "$A" --out "$C/threshold_sweep_${TAG}_merged.json" || exit 1

step "threshold sweep with idle policies, each space at its own cap"
"$PY" threshold_sweep.py --points "$C/points_${TAG}_ts100.npz" --acc "$A" --latency "$C/latency_${TAG}_ts100.npz" \
    --fmax 100 --out "$C/threshold_sweep_${TAG}_ts100.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-220
"$PY" threshold_sweep.py --points "$C/points_${TAG}_ts125.npz" --acc "$A" --latency "$C/latency_${TAG}_ts125.npz" \
    --fmax 125 --out "$C/threshold_sweep_${TAG}_ts125.json" | grep -E "^== margin|cont vs" | cut -c1-220
echo "== done $(date +%T)"
