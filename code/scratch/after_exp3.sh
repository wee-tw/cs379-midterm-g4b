#!/bin/bash
# After the main accuracy queue (exp3): accuracy model, constrained search, pareto,
# proxy analysis, figures, then 30-epoch QAT of the selected candidate configurations.
S="C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
PY="$S/venv_finn010/Scripts/python.exe"
CODE="D:/115-1/embedded sys midterm report/code"
export PYTHONWARNINGS=ignore
cd "$CODE"
# wait until every sensitivity and validation run has its log
until [ "$(ls $S/exp3/sens_g*_w*a*.json 2>/dev/null | wc -l)" -ge 32 ] && \
      [ "$(ls $S/exp3/val_*.json 2>/dev/null | wc -l)" -ge 8 ]; do sleep 30; done
echo "== accuracy model $(date)"
"$PY" acc_model.py --runs "$S/exp3" --base sens_base --reference uniform_model --margin 1.0 \
    --check "val_*" --out "$S/cost/acc_model.json" | tee "$S/cost/acc_model.log"
echo "== constrained search $(date)"
"$PY" energy_search.py search --points "$S/cost/points.npz" --acc "$S/cost/acc_model.json" \
    --out "$S/cost/search_acc.json" > "$S/cost/search_acc.log" 2>&1
echo "== pareto $(date)"
"$PY" energy_search.py pareto --points "$S/cost/points.npz" --acc "$S/cost/acc_model.json" \
    --out "$S/cost/pareto.json" --lo 74 --hi 86 --step 1 > "$S/cost/pareto.log" 2>&1
echo "== proxy with accuracy $(date)"
"$PY" analysis_proxy.py --points "$S/cost/points.npz" --acc "$S/cost/acc_model.json" \
    --out "$S/cost/proxy_acc.json" > "$S/cost/proxy_acc.log" 2>&1
echo "== figures $(date)"
"$PY" make_figures.py --runs "$S/exp3" --recipe "$S/exp2" "$S/exp3" --points "$S/cost/points.npz" \
    --acc "$S/cost/acc_model.json" --out "$CODE/results/figs" > "$S/cost/figs2.log" 2>&1
echo "== candidate QAT $(date)"
"$PY" -I - <<PYEOF
import json
S = r"$S"
res = json.load(open(S + "/cost/search_acc.json"))["nominal"]
cands = {}
def add(tag, d):
    if d: cands.setdefault(tuple(d["bits"]), []).append(tag)
for k, d in res["continuous"].items(): add("cont_" + k, d)
for R, r in res["fixed_rate"].items():
    for k, d in r.items(): add(f"fix{R}_" + k, d)
data = S + "/cifar_hf/cifar10.npz"
jobs = []
for bits, tags in cands.items():
    name = "cand_" + "_".join(map(str, bits))
    jobs.append({"name": name, "tags": tags, "args": ["--model", "brevitas", "--bits"] + [str(b) for b in bits] + [
        "--recipe", "lowbit", "--epochs", "30", "--lr", "0.05", "--batch", "256",
        "--init", S + "/exp2/finnfloat.pt", "--teacher", S + "/runs/float_rn8.pt", "--data", data]})
json.dump(jobs, open(S + "/exp4_jobs.json", "w"), indent=1)
print(len(jobs), "candidate configs:", [(j["name"], j["tags"]) for j in jobs])
PYEOF
# only 2 trainings may share the GPU and the 16 GB of RAM: wait for the exp3 queue to end
until grep -q "name, final_acc" "$S/exp3/queue.log" 2>/dev/null; do sleep 30; done
echo "== exp3 queue finished, starting candidates $(date)"
"$PY" run_queue.py "$S/exp4_jobs.json" --out "$S/exp4" --parallel 2 > "$S/exp4_queue.log" 2>&1
tail -20 "$S/exp4_queue.log"
echo "== done $(date)"
