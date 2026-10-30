#!/bin/bash
# After a board session: fit, then rerun the ResNet-8 search with the measured parameters.
#   bash postprocess.sh <session_dir> <meter_log> [extra analyze_session.py options]
# e.g. bash postprocess.sh sessions/20261030_160000 meter.txt \
#          --time-col timestamp --volt-col voltage_V --amp-col current_A
# Needs: est/util_cnv_est.json and est/<model>/calibration.json (FINN estimates of the
# finn-examples bitstreams), cost/v0/points.npz and cost/v0/acc_model.json (copied next to code/).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$(dirname "$HERE")"
PY="${PY:-python}"
SESSION="$1"; METER="$2"; shift 2
"$PY" "$HERE/analyze_session.py" "$SESSION" --meter-csv "$METER" \
    --util "$HERE/est/util_cnv_est.json" --fe-est "$HERE/est"     --pred "$HERE/predictions/predictions.json" "$@"
P="$SESSION/power_params.json"
[ -f "$P" ] || { echo "no global fit (need >= 2 bitstreams with idle and continuous data)"; exit 0; }
cd "$CODE"
"$PY" energy_search.py search --points cost/v0/points.npz --acc cost/v0/acc_model.json \
    --power-json "$P" --out "$SESSION/search_measured.json" > "$SESSION/search_measured.log"
"$PY" energy_search.py idle --points cost/v0/points.npz --acc cost/v0/acc_model.json \
    --power-json "$P" --latency cost/v0/latency.npz --lat-scale 0.6024 \
    --out "$SESSION/idle_measured.json" > "$SESSION/idle_measured.log"
"$PY" idle_summary.py "$SESSION/idle_measured.json" | grep z2_measured
"$PY" latency_summary.py "$SESSION/idle_measured.json" | grep -A5 z2_measured
"$PY" make_figures.py --idle "$SESSION/idle_measured.json" --latency "$SESSION/idle_measured.json" \
    --out "$SESSION"
echo "done: $SESSION (summary.md, power_params.json, *_measured.json, figures)"
