"""Jobs for run_queue.py: the candidate fine-tune used for every verified pick (exp9-exp13):
W4A4 base (exp3/sens_base0.pt) -> 30 epochs, lowbit, KD from the float teacher, lr 0.02,
batch 256. Usage: make_qat_jobs.py <out.json> <epochs> <seeds e.g. 0,1,2> <bits> [<bits> ...]
Seed 0 is named ft30_cand_<bits> (as in exp9), others ft30_cand_<bits>_s<seed>."""
import json
import sys

S = "C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad"
out, epochs, seeds, configs = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
jobs = []
for b in configs:
    assert len(b) == 8 and set(b) <= set("248"), b
    for s in seeds:
        jobs.append({"name": f"ft{epochs}_cand_{b}" + ("" if s == "0" else f"_s{s}"),
                     "args": ["--model", "brevitas", "--bits", *list(b), "--recipe", "lowbit",
                              "--epochs", epochs, "--lr", "0.02", "--batch", "256",
                              "--init", S + "/exp3/sens_base0.pt", "--data", S + "/cifar_hf/cifar10.npz",
                              "--teacher", S + "/runs/float_rn8.pt", "--seed", s]})
json.dump(jobs, open(out, "w"), indent=1)
print(len(jobs), "jobs ->", out)
