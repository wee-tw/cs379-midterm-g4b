#!/bin/bash
# Timing-aware main results (notes 12.35-12.36). Usage:
#   run_ts_v3.sh <raw tables from cost_tables.py --ts-refold (lut, dspc, dsp4c)> <tag> <rtl timing json> [limit]
# lut_finn comes unchanged from tables.json. The raw tables go through the validated calibration
# chain (calib_chain.sh reproduces tables_calv2.json exactly); timing_safe.py puts every record
# in a clock class (an FCLK0 setting 1000/n MHz, 125 down to 50) from the measured node timing; each class is
# searched with its clocks capped and the selections are merged.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
C="$S/cost"; A="$C/acc_model_3rep.json"; RAW="$1"; TAG="tsr_$2"; RTL="$3"; LIM="${4:+--limit $4}"
CLASSES="125 111.11 100 90.91 83.33 76.92 71.43 66.67 50"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
cd "$CODE" || exit 1
step() { echo "== $1 $(date +%T)"; }
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
need "$RAW"; need "$RTL"

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
"$PY" timing_safe.py tables --tables "$C/tables_calv2_$TAG.json" --rtl-timing "$RTL" --prefix "$C/tables_calv2_$TAG" || exit 1
for c in $CLASSES; do need "$C/tables_calv2_${TAG}_f$c.json"; done

step "points (all classes in parallel)"
for c in $CLASSES; do
  "$PY" energy_search.py points --tables "$C/tables_calv2_${TAG}_f$c.json" $LIM \
      --out "$C/points_${TAG}_f$c.npz" > "$C/points_${TAG}_f$c.log" 2>&1 &
done
wait
for c in $CLASSES; do need "$C/points_${TAG}_f$c.npz"; echo "f$c: $(tail -1 "$C/points_${TAG}_f$c.log" | grep -o '[0-9]* points')"; done

step "merged sweep"
PTS=""; for c in $CLASSES; do PTS="$PTS $c=$C/points_${TAG}_f$c.npz"; done
"$PY" timing_safe.py sweep --acc "$A" --out "$C/threshold_sweep_${TAG}_merged.json" --points $PTS || exit 1

step "idle policies at 100 MHz (designs of class >= 100)"
"$PY" latency_check.py points --tables "$C/tables_calv2_${TAG}_f100.json" --points "$C/points_${TAG}_f100.npz" \
    --out "$C/latency_${TAG}_f100.npz" | tail -1
need "$C/latency_${TAG}_f100.npz"
"$PY" threshold_sweep.py --points "$C/points_${TAG}_f100.npz" --acc "$A" --latency "$C/latency_${TAG}_f100.npz" \
    --fmax 100 --out "$C/threshold_sweep_${TAG}_f100.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-220
echo "== done $(date +%T)"
