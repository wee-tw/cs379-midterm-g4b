#!/bin/bash
# Fourth batch (12.33): the fastest uniform W4A4 design with its fewest-LUT variants, to check that
# it does not fit (B1 at margins 2-3). Waits for the third batch.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
until grep -q "== done" "$S/after_vd3.log" 2>/dev/null; do sleep 30; done
echo "== generate $(date +%T)"
"$PY" verify_designs.py generate --tables "$S/cost/tables_calv2.json" --out D:/vd4 --finn-root "$F" \
    --designs-json "$S/designs_vd4.json" 2>&1 | grep -v "outside fpgadataflow" | tail -3
echo "== run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd4 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 3 --workdir D:/vw4 > "$S/verify_vd4.log" 2>&1
tail -2 "$S/verify_vd4.log"
"$PY" verify_designs.py report --out D:/vd4 --designs-json "$S/designs_vd4.json" > "$S/verify_vd4_report.log" 2>&1
echo "== done $(date +%T)"
