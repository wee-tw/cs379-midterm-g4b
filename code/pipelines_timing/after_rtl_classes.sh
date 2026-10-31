#!/bin/bash
# After the constrained RTL timing rerun: timing classes of the refolded table with the
# constrained RTL periods, at routing margins 0, 10 % and 20 % (run_ts_classes.sh, tested).
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
until [ -s "$S/rtl_timing/rtl_all_constrained.json" ]; do sleep 30; done
for m in 0 0.1 0.2; do
  tag="tsr_v2c_m$m"
  echo "######## margin $m ($tag) $(date +%T)"
  bash "$S/run_ts_classes.sh" "$S/cost/tables_calv2_tsr_v2.json" "$tag" "$S/rtl_timing/rtl_all_constrained.json" 3 "$m" \
      | grep -E "records by class|without a measurement|^f[0-9]|^== margin|nominal|z2_anchor|MISSING|done"
done
