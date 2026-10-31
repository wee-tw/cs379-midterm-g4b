#!/bin/bash
# After exp5: seed variance, pairwise W2 correction of the accuracy model, and the search,
# idle-policy and threshold-sweep analyses repeated with the corrected model.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
cd "$CODE"
until grep -q "== done" "$S/after_extra2.log" 2>/dev/null; do sleep 60; done
echo "== seed variance $(date)"
"$PY" -I - "$S" <<'PYEOF'
import json, os, statistics, sys
S = sys.argv[1]
def acc(d, n):
    f = os.path.join(S, d, n + ".json")
    return json.load(open(f))["history"][-1]["test_acc"] if os.path.exists(f) else None
groups = {"control W4A4 (8-ep fine-tune)": [("exp3", "sens_base")] + [("exp5", f"seed_base_s{s}") for s in (1, 2, 3)],
          "G1 W2A4": [("exp3", "sens_g1_w2a4")] + [("exp5", f"seed_g1_w2a4_s{s}") for s in (1, 2)],
          "G3 W8A4": [("exp3", "sens_g3_w8a4")] + [("exp5", f"seed_g3_w8a4_s{s}") for s in (1, 2)],
          "G4 W4A2": [("exp3", "sens_g4_w4a2"), ("exp5", "seed_g4_w4a2_s1")],
          "uniform W4A4 (30-ep)": [("exp3", "sens_base0"), ("exp5", "seed_b1_w4a4_s1")],
          "cand 44444488 (30-ep)": [("exp4", "cand_4_4_4_4_4_4_8_8"), ("exp5", "seed_cand_44444488_s1")],
          "cand 44444488 (60-ep)": [("exp5", "long_cand_44444488_e60")]}
out = {}
for k, runs in groups.items():
    v = [a for a in (acc(d, n) for d, n in runs) if a is not None]
    out[k] = {"acc": v, "mean": round(statistics.mean(v), 2) if v else None,
              "sd": round(statistics.stdev(v), 2) if len(v) > 1 else None}
    print(f"{k:30s} {v} mean {out[k]['mean']} sd {out[k]['sd']}")
json.dump(out, open(os.path.join(S, "cost", "seed_variance.json"), "w"), indent=1)
PYEOF
echo "== pairwise accuracy correction $(date)"
"$PY" acc_interactions.py --acc "$S/cost/acc_model.json" --runs "$S/exp3" "$S/exp5" \
    --out "$S/cost/acc_model_int.json" | tee "$S/cost/acc_model_int.log"
echo "== search / idle / sweep with the corrected model $(date)"
"$PY" energy_search.py search --points "$S/cost/points.npz" --acc "$S/cost/acc_model_int.json" \
    --out "$S/cost/search_acc_int.json" > "$S/cost/search_acc_int.log" 2>&1
"$PY" energy_search.py idle --points "$S/cost/points.npz" --acc "$S/cost/acc_model_int.json" \
    --out "$S/cost/idle_acc_int.json" > "$S/cost/idle_acc_int.log" 2>&1
"$PY" idle_summary.py "$S/cost/idle_acc_int.json" >> "$S/cost/idle_acc_int.log" 2>&1
"$PY" threshold_sweep.py --points "$S/cost/points.npz" --acc "$S/cost/acc_model_int.json" \
    --out "$S/cost/threshold_sweep_int.json" > "$S/cost/threshold_sweep_int.log" 2>&1
grep -E "^== margin|cont vs" "$S/cost/threshold_sweep_int.log" | cut -c1-170
grep "cont gap" "$S/cost/idle_acc_int.log"
echo "== done $(date)"
