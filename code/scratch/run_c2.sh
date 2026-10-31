#!/bin/bash
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
until grep -q "^saved" "$S/cost/tables_c2.log" 2>/dev/null; do sleep 20; done
bash "$S/after_c2.sh" lo > "$S/after_c2_lo.log" 2>&1 &
bash "$S/after_c2.sh" hi > "$S/after_c2_hi.log" 2>&1 &
wait
echo "both done $(date +%T)"
