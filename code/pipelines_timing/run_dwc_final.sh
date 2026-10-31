#!/bin/bash
# DWCs of the final designs (A', B1, B2): generate FINN's code, synthesize (Vivado / Vitis HLS, 8 ns or
# each design's clock), then the per-design totals and the energy with DWCs. notes 12.47.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
PY="$S/venv_finn010/Scripts/python.exe"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$CODE/pipelines_timing" || exit 1
echo "== generate $(date +%T)"
"$PY" dwc_final.py generate --designs "$S/vdw_designs.json" --out D:/vdw 2>&1 | grep -v -i "warn"
[ -s D:/vdw/cases.json ] || { echo "generate did not finish"; exit 1; }
echo "== synthesize $(date +%T)"
cd "$CODE" || exit 1
"$PY" verify_designs.py run --out D:/vdw --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 4 --workdir D:/vww --timing 2>&1 | grep -v -i "warn"
echo "== report $(date +%T)"
cd "$CODE/pipelines_timing" || exit 1
P="$S/cost/points_tsr_v1218"
"$PY" dwc_final.py report --designs "$S/vdw_designs.json" --out D:/vdw \
    --table "../../results/search/main_table_tsr_v1218_m0.json" \
    --points 125=${P}_f125.npz 111.11=${P}_f111.11.npz 100=${P}_f100.npz 2>&1 | grep -v -i "warn"
echo "== dwc all done $(date +%T)"
