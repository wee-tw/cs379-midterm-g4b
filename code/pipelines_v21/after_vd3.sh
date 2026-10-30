#!/bin/bash
# Third design-verification batch (12.33): the v2.1 picks not yet synthesized, with the v2.1 tables'
# variant choices. Waits for the v2.1 reruns (memory).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE"
until grep -q "== all done" "$S/master_v21.log" 2>/dev/null; do sleep 30; done
echo "== generate $(date +%T)"
"$PY" verify_designs.py generate --tables "$S/cost/tables_calv2.json" --out D:/vd3 --finn-root "$F" \
    --designs-json "$S/designs_vd3.json" 2>&1 | grep -v "outside fpgadataflow" | tail -3
echo "== run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd3 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 3 --workdir D:/vw3 > "$S/verify_vd3.log" 2>&1
tail -2 "$S/verify_vd3.log"
"$PY" verify_designs.py report --out D:/vd3 --designs-json "$S/designs_vd3.json" > "$S/verify_vd3_report.log" 2>&1
"$PY" "$S/check_v2_designs.py" "$S" --v2-choices D:/vd3 2>&1 | tail -5
echo "== done $(date +%T)"
