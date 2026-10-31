#!/bin/bash
# Corrected design spaces from tables_c2 (MVAU_rtl DSPs as instantiated; LUTs bracketed):
# lo = RTL estimate plus the LUTRAM it omits, hi = the DSP-based HLS MVAU estimate.
# Both keep lut + lut_finn unchanged. Usage: after_c2.sh lo|hi
V="$1"
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$CODE"
"$PY" - "$S" "$V" <<'PYEOF'
import json, sys
S, V = sys.argv[1], sys.argv[2]
a = json.load(open(S + "/cost/tables.json"))
c = json.load(open(S + "/cost/tables_c2.json"))
keep = {k: v for k, v in a["table"].items() if k.split("|")[0] in ("lut", "lut_finn")}
for k, rec in c["table"].items():
    for n in rec["nodes"]:
        if V == "hi" and "LUT_hi" in n["auto"]:
            n["auto"]["LUT"] = n["auto"]["LUT_hi"]
            for v in n.get("variants", []):
                v["LUT"] = v["LUT_hi"]
    keep[k] = rec
json.dump({"T_GRID": a["T_GRID"], "modes": ["lut", "lut_finn"] + c["modes"], "table": keep,
           "mismatches": c["mismatches"]}, open(f"{S}/cost/tables_corr_{V}.json", "w"))
print(V, "records", len(keep))
PYEOF
T="$S/cost/tables_corr_$V.json"; P="$S/cost/points_corr_$V.npz"; L="$S/cost/latency_corr_$V.npz"
"$PY" energy_search.py points --tables "$T" --out "$P" > "$S/cost/points_corr_$V.log" 2>&1
tail -1 "$S/cost/points_corr_$V.log"
"$PY" latency_check.py points --tables "$T" --points "$P" --out "$L"
echo "== [$V] main search (3-replicate model, margin 1)"
"$PY" energy_search.py search --points "$P" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/search_corr_$V.json" > "$S/cost/search_corr_$V.log" 2>&1
grep -A6 "power params: nominal" "$S/cost/search_corr_$V.log" | head -7 | cut -c1-200
echo "== [$V] threshold sweep with latency-aware policies"
"$PY" threshold_sweep.py --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --out "$S/cost/threshold_sweep_corr_$V.json" | grep -E "^== margin|cont vs|jit:" | cut -c1-200
echo "== [$V] latency-aware idle analysis"
"$PY" energy_search.py idle --points "$P" --acc "$S/cost/acc_model_3rep.json" --latency "$L" \
    --lat-scale 0.6024 --out "$S/cost/idle_lat_corr_$V.json" > "$S/cost/idle_lat_corr_$V.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_corr_$V.json" | grep -E "^== nominal|^== z2|R=  300|R= 3000" | cut -c1-240
echo "== [$V] proxies"
"$PY" proxy_finn.py --points "$P" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/proxy_finn_corr_$V.json" | grep -E "margin|tau\(BOPs |tau\(weight bits " | cut -c1-260
echo "== [$V] key designs"
"$PY" "$S/compare_designs.py" "$P" 44444448 44444488 44242448 84242448 44224448 42442448 48484848
echo "== [$V] done $(date +%T)"
