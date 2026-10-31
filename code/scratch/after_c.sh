#!/bin/bash
# After the corrected DSP-mode cost tables: a design space with lut + lut_finn (unchanged)
# and the two corrected DSP modes (FINN's own dsp/dsp4 estimates left out), then the H1/H3
# sweeps, the main search, the latency-aware idle analysis and the proxy check on it.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
F="C:/Users/wee/AppData/Local/Temp/claude/C--SPB-Data--claude-spares-spare-5cUE7w/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/repos/finn-v0.10.1"
export PYTHONPATH="$F/src;$CODE" FINN_ROOT="$F" OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$CODE"
"$PY" - "$S" <<'PYEOF'
import json, sys
S = sys.argv[1]
a = json.load(open(S + "/cost/tables.json"))
c = json.load(open(S + "/cost/tables_c.json"))
assert a["T_GRID"] == c["T_GRID"]
keep = {k: v for k, v in a["table"].items() if k.split("|")[0] in ("lut", "lut_finn")}
keep.update(c["table"])
out = {"T_GRID": a["T_GRID"], "modes": ["lut", "lut_finn"] + c["modes"], "table": keep,
       "mismatches": c["mismatches"]}
json.dump(out, open(S + "/cost/tables_corr.json", "w"))
print("corrected design space: records", len(keep), "modes", out["modes"],
      "mismatches", len(c["mismatches"]))
PYEOF
"$PY" energy_search.py points --tables "$S/cost/tables_corr.json" --out "$S/cost/points_corr.npz" > "$S/cost/points_corr.log" 2>&1
tail -1 "$S/cost/points_corr.log"
"$PY" latency_check.py points --tables "$S/cost/tables_corr.json" --points "$S/cost/points_corr.npz" --out "$S/cost/latency_corr.npz"
echo "== main search, margin 1 (3-replicate model)"
"$PY" energy_search.py search --points "$S/cost/points_corr.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/search_corr.json" > "$S/cost/search_corr.log" 2>&1
grep -E "power params|A_prime|B1_uniform|B2_proxy|B3_seq|B4_lat" "$S/cost/search_corr.log" | head -14 | cut -c1-200
echo "== threshold sweep with latency-aware policies"
"$PY" threshold_sweep.py --points "$S/cost/points_corr.npz" --acc "$S/cost/acc_model_3rep.json" \
    --latency "$S/cost/latency_corr.npz" --out "$S/cost/threshold_sweep_corr.json" \
    | grep -E "^== margin|cont vs|jit:" | cut -c1-200
echo "== latency-aware idle analysis"
"$PY" energy_search.py idle --points "$S/cost/points_corr.npz" --acc "$S/cost/acc_model_3rep.json" \
    --latency "$S/cost/latency_corr.npz" --lat-scale 0.6024 --out "$S/cost/idle_lat_corr.json" \
    > "$S/cost/idle_lat_corr.log" 2>&1
"$PY" latency_summary.py "$S/cost/idle_lat_corr.json" | cut -c1-240
echo "== proxies"
"$PY" proxy_finn.py --points "$S/cost/points_corr.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/proxy_finn_corr.json" | grep -E "margin|tau\(BOPs " | cut -c1-260
echo "== key designs"
"$PY" "$S/compare_designs.py" "$S/cost/points_corr.npz" 44444488 44444448 84242448 44242448 84222448 42442448 48484848
echo "== done $(date +%T)"
