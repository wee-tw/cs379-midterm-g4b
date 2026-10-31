#!/bin/bash
# B2 designs (FINN default folding, resType lut, 100 MHz) of margins 2-8, node by node, for their
# Vivado dynamic power next to the A' designs (notes 12.43). Same flow as verify_b2lut (margin 1).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE" || exit 1
echo "== generate $(date +%T)"
"$PY" verify_designs.py generate --tables "$S/cost/tables_calv2_tsr_v1218.json" --out D:/vd7 --finn-root "$F" \
    --designs-json "$S/vd7_designs.json" 2>&1 | grep -v "outside fpgadataflow" | tail -6
[ -s D:/vd7/cases.json ] || { echo "generate did not finish, stopping"; exit 1; }
echo "== smoke run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd7 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 1 --workdir D:/vw7 --timing --only d0_00_ > "$S/verify_vd7_smoke.log" 2>&1
cat "$S/verify_vd7_smoke.log" | cut -c1-200
grep -q "rc=0" "$S/verify_vd7_smoke.log" || { echo "smoke run did not pass, stopping"; exit 1; }
echo "== run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd7 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 2 --workdir D:/vw7 --timing > "$S/verify_vd7.log" 2>&1
tail -2 "$S/verify_vd7.log"
"$PY" verify_designs.py report --out D:/vd7 --designs-json "$S/vd7_designs.json" > "$S/verify_vd7_report.log" 2>&1
grep "^==" "$S/verify_vd7_report.log" | sed 's/| not synthesized.*//'
echo "== vd7 done $(date +%T)"
