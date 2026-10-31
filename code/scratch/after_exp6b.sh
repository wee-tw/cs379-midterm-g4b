#!/bin/bash
# After exp6 (no training running): FINN estimates of the finn-examples TFC-W1A1 and KWS
# bitstreams, so the board fit has bitstreams with low and high utilisation.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
EST="$CODE/board/est"
FOLD="$CODE/board/finn_examples_folding"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_exp6.log" 2>/dev/null; do sleep 60; done
echo "== TFC-W1A1 estimate $(date)"
"$PY" -u calib_cnv_mlperf.py --net tfc_1w1a --clk-ns 5 --out "$S/fe_util/tfc-w1a1" \
    --folding "$FOLD/tfc-w1a1_folding_config.json" \
    --specialize "$FOLD/tfc-w1a1_specialize_layers.json" > "$S/fe_util_tfc-w1a1.log" 2>&1
tail -12 "$S/fe_util_tfc-w1a1.log"
echo "== KWS estimate $(date)"
"$PY" -u estimate_onnx.py "$S/dl_fe/kws_model/MLP_W3A3_python_speech_features_pre-processing_QONNX_opset-11.onnx" \
    --folding "$FOLD/kwsmlp-w3a3_folding_config.json" --specialize "$FOLD/kwsmlp-w3a3_specialize_layers.json" \
    --clk-ns 10 --topk --out "$S/fe_util/kwsmlp-w3a3" > "$S/fe_util_kws.log" 2>&1
tail -12 "$S/fe_util_kws.log"
for m in tfc-w1a1 kwsmlp-w3a3; do
  [ -f "$S/fe_util/$m/calibration.json" ] && mkdir -p "$EST/$m" && cp "$S/fe_util/$m/calibration.json" "$EST/$m/"
done
"$PY" -I - "$EST" <<'PYEOF'
import json, os, sys
E = sys.argv[1]
dev = {"LUT": 53200, "BRAM_18K": 280, "DSP": 220}
u = json.load(open(os.path.join(E, "util_cnv_est.json")))
for m in ("tfc-w1a1", "kwsmlp-w3a3"):
    f = os.path.join(E, m, "calibration.json")
    if os.path.exists(f):
        e = json.load(open(f))["estimate_finn_v0101"]
        u[m] = round(e["LUT"] / dev["LUT"] + 0.25 * e["BRAM_18K"] / dev["BRAM_18K"]
                     + 0.25 * e["DSP"] / dev["DSP"], 4)
json.dump(u, open(os.path.join(E, "util_cnv_est.json"), "w"), indent=1)
print(u)
PYEOF
echo "== done $(date)"
