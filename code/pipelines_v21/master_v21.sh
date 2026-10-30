#!/bin/bash
# v2.1 reruns: main pipeline in parallel with the harm study, then sensitivity, then figures.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
bash "$S/after_calv2.sh" > "$S/after_calv2.log" 2>&1 &
bash "$S/after_harm_v2.sh" > "$S/after_harm_v2.log" 2>&1
wait
bash "$S/after_sens_v2.sh" > "$S/after_sens_v2.log" 2>&1
bash "$S/after_figs_v2.sh" > "$S/after_figs_v2.log" 2>&1
echo "== all done $(date +%T)"
