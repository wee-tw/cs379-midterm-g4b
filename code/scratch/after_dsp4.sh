#!/bin/bash
# After the DSP-aware cost tables: merge, design points with 4 modes, latency, then the
# H3 threshold sweep and the proxy check on the larger design space.
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
b = json.load(open(S + "/cost/tables_dsp4.json"))
assert a["T_GRID"] == b["T_GRID"]
a["table"].update(b["table"])
a["modes"] = list(a["modes"]) + [m for m in b["modes"] if m not in a["modes"]]
a["mismatches"] += b["mismatches"]
json.dump(a, open(S + "/cost/tables_all.json", "w"))
print("merged records", len(a["table"]), "modes", a["modes"], "mismatches", len(a["mismatches"]))
PYEOF
"$PY" energy_search.py points --tables "$S/cost/tables_all.json" --out "$S/cost/points_dsp4.npz" > "$S/cost/points_dsp4.log" 2>&1
tail -1 "$S/cost/points_dsp4.log"
"$PY" latency_check.py points --tables "$S/cost/tables_all.json" --points "$S/cost/points_dsp4.npz" --out "$S/cost/latency_dsp4.npz"
echo "== threshold sweep, DSP-aware design space (3-replicate accuracy model)"
"$PY" threshold_sweep.py --points "$S/cost/points_dsp4.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/threshold_sweep_dsp4.json" | grep -E "^== margin|cont vs" | cut -c1-200
echo "== proxies on the DSP-aware design space"
"$PY" proxy_finn.py --points "$S/cost/points_dsp4.npz" --acc "$S/cost/acc_model_3rep.json" \
    --out "$S/cost/proxy_finn_dsp4.json" | grep -E "margin|tau\(BOPs " | cut -c1-260
echo "== key designs"
"$PY" "$S/compare_designs.py" "$S/cost/points_dsp4.npz" 84242448 44242448 84222448 42442448 44444448 44444488 48484848
echo "== done $(date +%T)"
