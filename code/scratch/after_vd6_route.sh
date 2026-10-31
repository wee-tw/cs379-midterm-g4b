#!/bin/bash
# Route check (8 ns, Vivado default implementation) of the critical nodes of the final A' designs,
# each as soon as vd6 has synthesized it: RTL MVUs at SIMD 9-12 with PE up to 16 and wide
# unfolded (SF = 1) HLS MVUs that were not routed before.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
cd "D:/115-1/embedded sys midterm report/code" || exit 1
export PYTHONIOENCODING=utf-8
for c in d0_39_MVAU_rtl_5 d0_34_MVAU_rtl_4 d0_02_MVAU_rtl_0 d0_25_MVAU_hls_2 d3_07_MVAU_hls_1 d3_02_MVAU_hls_0; do
  case $c in
    *rtl*) until [ -s "D:/vw6/$c/vivado_timing.rpt" ]; do sleep 30; done ;;
    *hls*) until [ -s "D:/vd6/$c/export_syn.rpt" ] && ls D:/vw6/$c/project_*/sol1/impl/verilog/project.runs/bd_0_hls_inst_0_synth_1/*.dcp >/dev/null 2>&1; do sleep 30; done ;;
  esac
  "$S/venv_finn010/Scripts/python.exe" route_timing.py --vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat --period 8 \
      --workdir D:/vw6 --cases "$c" --out-dir D:/vr6 --out "$S/route/final_$c.json" | cut -c1-200
done
echo "== route vd6 done $(date +%T)"
