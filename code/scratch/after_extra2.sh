#!/bin/bash
# 10/30 night, after after_extra.sh: FINN estimates of the three finn-examples CNV bitstreams
# (U for the board-measurement fit), then exp5 (seed variance, interactions, long run).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_extra.log" 2>/dev/null; do sleep 60; done
echo "== finn-examples CNV estimates $(date)"
for net in cnv_1w1a cnv_1w2a cnv_2w2a; do
  m="cnv-w${net:4:1}a${net:6:1}"
  "$PY" -u calib_cnv_mlperf.py --net $net --clk-ns 5 --out "$S/fe_util/$m" \
      --folding "board/finn_examples_folding/${m}_folding_config.json" \
      --specialize "board/finn_examples_folding/${m}_specialize_layers.json" \
      > "$S/fe_util_$m.log" 2>&1
  tail -12 "$S/fe_util_$m.log"
done
"$PY" -I - "$S" <<'PYEOF'
import json, os, sys
S = sys.argv[1]
dev = {"LUT": 53200, "BRAM_18K": 280, "DSP": 220}
out = {}
for m in ("cnv-w1a1", "cnv-w1a2", "cnv-w2a2"):
    f = os.path.join(S, "fe_util", m, "calibration.json")
    if not os.path.exists(f):
        continue
    e = json.load(open(f))["estimate_finn_v0101"]
    out[m] = round(e["LUT"] / dev["LUT"] + 0.25 * e["BRAM_18K"] / dev["BRAM_18K"]
                   + 0.25 * e["DSP"] / dev["DSP"], 4)
    print(m, e, "U_est", out[m])
json.dump(out, open(os.path.join(S, "util_cnv_est.json"), "w"), indent=1)
PYEOF
echo "== exp5 $(date)"
"$PY" run_queue.py "$S/exp5_jobs.json" --out "$S/exp5" --parallel 2 > "$S/exp5_queue.log" 2>&1
tail -20 "$S/exp5_queue.log"
echo "== done $(date)"
