#!/bin/bash
# Node-by-node synthesis of the designs picked with the routing-checked refold (RTL <= 12,
# HLS <= 18): verified A' and B1 of every margin (make_vd6_designs.py), generated as the table
# was built and checked against its cycles, then synthesized with the clock (D:/vd6).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
until [ -s "$S/cost/main_table_tsr_v1218_m0.json" ] || grep -q "failed" "$S/after_v5.log" 2>/dev/null; do sleep 30; done
[ -s "$S/cost/main_table_tsr_v1218_m0.json" ] || { echo "no main table, stopping"; exit 1; }
cd "$CODE" || exit 1
"$PY" "$S/make_vd6_designs.py" "$S/cost/main_table_tsr_v1218_m0.json" "$S/vd6_designs.json"
echo "== generate $(date +%T)"
"$PY" verify_designs.py generate --tables "$S/cost/tables_calv2_tsr_v1218.json" --out D:/vd6 --finn-root "$F" \
    --designs-json "$S/vd6_designs.json" --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18 2>&1 \
    | grep -v "outside fpgadataflow" | tail -12
[ -s D:/vd6/cases.json ] || { echo "generate did not finish, stopping"; exit 1; }
echo "== smoke run: the first MVU of the first design $(date +%T)"
"$PY" verify_designs.py run --out D:/vd6 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat     --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1"     --jobs 1 --workdir D:/vw6 --timing --only d0_02_ > "$S/verify_vd6_smoke.log" 2>&1
cat "$S/verify_vd6_smoke.log" | cut -c1-200
grep -q "rc=0" "$S/verify_vd6_smoke.log" || { echo "smoke run did not pass, stopping"; exit 1; }
echo "== run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd6 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 3 --workdir D:/vw6 --timing > "$S/verify_vd6.log" 2>&1
tail -2 "$S/verify_vd6.log"
"$PY" verify_designs.py report --out D:/vd6 --designs-json "$S/vd6_designs.json" > "$S/verify_vd6_report.log" 2>&1
grep "^==" "$S/verify_vd6_report.log" | sed 's/| not synthesized.*//'
echo "== done $(date +%T)"
