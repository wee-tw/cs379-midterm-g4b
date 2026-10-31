#!/bin/bash
# New margin-5 A' after the 5-seed rule (24244448 dspc T14286, 125 MHz): node-by-node synthesis (D:/vd8),
# its width converters (added to D:/vdw as design 10), the narrowed-MW576 converters, and the FINN numerics.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
CODE="D:/115-1/embedded sys midterm report/code"; PY="$S/venv_finn010/Scripts/python.exe"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
VIV="--vivado D:/Xilinx/Vivado/2022.2/bin/vivado.bat --vitis-hls D:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat --finn-root $F --mvu-rtl $S/upstream/mvu_v0.10.1"
cd "$CODE" || exit 1
# FINN numerics in parallel (CPU only): 20-image smoke, then 500 images
( "$PY" check_finn_numerics.py --bits 2 4 2 4 4 4 4 8 --ckpt "$S/exp9/ft30_cand_24244448.pt" --npz "$S/cifar_hf/cifar10.npz" \
      --recipe lowbit --n 20 > "$S/numerics_m5_n20.log" 2>&1 && grep -q "hw .* 20/20" "$S/numerics_m5_n20.log" \
  && "$PY" check_finn_numerics.py --bits 2 4 2 4 4 4 4 8 --ckpt "$S/exp9/ft30_cand_24244448.pt" --npz "$S/cifar_hf/cifar10.npz" \
      --recipe lowbit --n 500 > "$S/numerics_m5_n500.log" 2>&1; echo "== numerics m5 finished rc=$? $(date +%T)" ) &
echo "== vd8 generate $(date +%T)"
"$PY" verify_designs.py generate --tables "$S/cost/tables_calv2_tsr_v1218.json" --out D:/vd8 --finn-root "$F" \
    --designs-json "$S/vd8_designs.json" --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18 2>&1 | grep -v "outside fpgadataflow" | tail -4
[ -s D:/vd8/cases.json ] || { echo "vd8 generate did not finish"; wait; exit 1; }
"$PY" verify_designs.py run --out D:/vd8 $VIV --jobs 1 --workdir D:/vw8 --timing --only d0_02_ > "$S/verify_vd8_smoke.log" 2>&1
grep -q "rc=0" "$S/verify_vd8_smoke.log" || { echo "vd8 smoke did not pass"; cat "$S/verify_vd8_smoke.log" | tail -5; wait; exit 1; }
echo "== vd8 run $(date +%T)"
"$PY" verify_designs.py run --out D:/vd8 $VIV --jobs 3 --workdir D:/vw8 --timing > "$S/verify_vd8.log" 2>&1
"$PY" verify_designs.py report --out D:/vd8 --designs-json "$S/vd8_designs.json" > "$S/verify_vd8_report.log" 2>&1
grep "^==" "$S/verify_vd8_report.log" | cut -c1-260
echo "== DWC generate (11 designs into D:/vdw) $(date +%T)"
cd "$CODE/pipelines_timing" || exit 1
"$PY" dwc_final.py generate --designs "$S/vdw_designs11.json" --out D:/vdw 2>&1 | grep -v -i "warn\|outside" | tail -2
"$PY" dwc_final.py generate --designs "$S/vdw_narrow_m5_designs.json" --out D:/vdw_narrow_m5 --narrow 576:64:64:3 2>&1 | grep -v -i "warn\|outside" | tail -2
cd "$CODE" || exit 1
"$PY" verify_designs.py run --out D:/vdw $VIV --jobs 3 --workdir D:/vww --timing > "$S/verify_vdw11.log" 2>&1
grep -c "rc=0" "$S/verify_vdw11.log"; grep "rc=" "$S/verify_vdw11.log" | grep -vc "rc=0"
echo "== waiting for numerics $(date +%T)"
wait
echo "== verify m5 all done $(date +%T)"
