#!/bin/bash
# Vivado / Vitis HLS 2022.2 audits (implementation_notes.md 12.27): HLS MVAU threshold styles
# (FINN-default folding first: it decides B2), then the relaxed-folding nodes, then the 159 RTL
# nodes out of context. Work copies go to short paths (Windows 260-character limit).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"
HLS="D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat"; VIV="D:/Xilinx/Vivado/2022.2/bin/vivado.bat"
cd "$CODE"
echo "== HLS, FINN-default folding $(date +%T)"
python -I audit_hls_thresholds.py run --out "$S/audit_hls_finn" --finn-root "$F" --vitis-hls "$HLS" --jobs 6 --workdir D:/hf
echo "== HLS, relaxed folding $(date +%T)"
python -I audit_hls_thresholds.py run --out "$S/audit_hls" --finn-root "$F" --vitis-hls "$HLS" --jobs 6 --workdir D:/ha
echo "== Vivado OOC, RTL nodes $(date +%T)"
python -I audit_rtl_ops.py vivado --out "$S/audit" --vivado "$VIV" --mvu-rtl "$S/upstream/mvu_v0.10.1" --jobs 6 --workdir D:/va
echo "== done $(date +%T)"
