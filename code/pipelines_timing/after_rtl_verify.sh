#!/bin/bash
# After the constrained RTL timing rerun (memory): synthesize every node of the 8 timing-aware
# designs (D:/vd5, generated with --ts-refold --ts-rtl-simd 27 and checked against the table's
# cycles; one refolded RTL MVU smoke-tested in D:/vt_test). Clock given to RTL synthesis.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
until [ -s "$S/rtl_timing/rtl_all_constrained.json" ]; do sleep 30; done
echo "== verify start $(date +%T)"
cd "$CODE" || exit 1
"$S/venv_finn010/Scripts/python.exe" verify_designs.py run --out D:/vd5 --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat \
    --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root "$F" --mvu-rtl "$S/upstream/mvu_v0.10.1" \
    --jobs 3 --workdir D:/vw5 --timing
echo "== verify done $(date +%T)"
"$S/venv_finn010/Scripts/python.exe" verify_designs.py report --out D:/vd5 --designs-json "$S/vd5_designs.json"
