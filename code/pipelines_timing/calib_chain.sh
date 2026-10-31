#!/bin/bash
# The calibration chain that turned the raw FINN cost tables into tables_calv2.json
# (after_c2.sh hi -> threshold_consistent -> calib_mvu_vivado -> calib_mvau_hls --override ->
# replay_bram -> calib_v2 --override). Usage: calib_chain.sh <lut tables> <dspc/dsp4c tables> <tag>
# writes $C/tables_<step>_<tag>.json for every step and ends with $C/tables_calv2_<tag>.json.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PYTHONUTF8=1
C="$S/cost"; LUTT="$1"; DSPT="$2"; TAG="$3"
cd "$CODE" || exit 1
need() { [ -s "$1" ] || { echo "MISSING $1, stopping"; exit 1; }; }
echo "== corr_hi (lut + lut_finn from $LUTT, dspc + dsp4c from $DSPT with LUT_hi)"
"$PY" - "$LUTT" "$DSPT" "$C/tables_corr_hi_$TAG.json" <<'PYEOF'
import json, sys
a, c = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
keep = {k: v for k, v in a["table"].items() if k.split("|")[0] in ("lut", "lut_finn")}
for k, rec in c["table"].items():
    if k.split("|")[0] not in ("dspc", "dsp4c"):
        continue
    for n in rec["nodes"]:
        if "LUT_hi" in n["auto"]:
            n["auto"]["LUT"] = n["auto"]["LUT_hi"]
            for v in n.get("variants", []):
                v["LUT"] = v["LUT_hi"]
    keep[k] = rec
json.dump({"T_GRID": a["T_GRID"], "modes": ["lut", "lut_finn", "dspc", "dsp4c"], "table": keep,
           "mismatches": c["mismatches"]}, open(sys.argv[3], "w"))
print("records", len(keep))
PYEOF
need "$C/tables_corr_hi_$TAG.json"
echo "== threshold_consistent"
"$PY" threshold_consistent.py --tables "$C/tables_corr_hi_$TAG.json" --out "$C/tables_thr_hi_$TAG.json" | tail -1
need "$C/tables_thr_hi_$TAG.json"
echo "== calib_mvu_vivado"
"$PY" calib_mvu_vivado.py tables --audit "$S/audit" --tables "$C/tables_thr_hi_$TAG.json" --out "$C/tables_viv_$TAG.json" | tail -1
need "$C/tables_viv_$TAG.json"
echo "== calib_mvau_hls --override"
"$PY" calib_mvau_hls.py tables --audit "$S/audit_hls_finn" "$S/audit_hls" --tables "$C/tables_viv_$TAG.json" \
    --out "$C/tables_viv_hlsov_$TAG.json" --override | tail -1
need "$C/tables_viv_hlsov_$TAG.json"
echo "== replay_bram"
"$PY" replay_bram.py tables --tables "$C/tables_viv_hlsov_$TAG.json" --out "$C/tables_calov_$TAG.json" | tail -1
need "$C/tables_calov_$TAG.json"
echo "== calib_v2 --override"
"$PY" calib_v2.py tables --override --fit "$C/calib_v2.json" --tables "$C/tables_calov_$TAG.json" --scratch "$S" \
    --out "$C/tables_calv2_$TAG.json" | tail -1
need "$C/tables_calv2_$TAG.json"
echo "== chain done $(date +%T)"
