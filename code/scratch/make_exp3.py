"""Job list for the main accuracy experiments (exp3).

  python make_exp3.py <recipe> <kd 0/1> <out dir>
"""
import itertools
import json
import os
import random
import sys

S = os.path.dirname(os.path.abspath(__file__))
recipe, kd, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
data = os.path.join(S, "cifar_hf", "cifar10.npz")
teacher = os.path.join(S, "runs", "float_rn8.pt")
fp_init = os.path.join(S, "exp2", "finnfloat.pt")
PAIRS = list(itertools.product((2, 4, 8), repeat=2))
E_LONG, E_SENS = 30, 8


def args(bits, init, epochs, lr):
    a = ["--model", "brevitas", "--bits"] + [str(b) for b in bits] + [
        "--recipe", recipe, "--epochs", str(epochs), "--lr", str(lr), "--batch", "256",
        "--init", init, "--data", data]
    return a + (["--teacher", teacher] if kd else [])


jobs = [{"name": "sens_base0", "args": args([4] * 8, fp_init, E_LONG, 0.05)}]
base_ckpt = "{out}/sens_base0.pt"
dep = ["sens_base0"]
sens = [{"name": "sens_base", "args": args([4] * 8, base_ckpt, E_SENS, 0.02), "after": dep}]
for g in range(4):
    for w, a in PAIRS:
        if (w, a) == (4, 4):
            continue
        bits = [4] * 8
        bits[2 * g], bits[2 * g + 1] = w, a
        sens.append({"name": f"sens_g{g + 1}_w{w}a{a}", "args": args(bits, base_ckpt, E_SENS, 0.02),
                     "after": dep})
rng = random.Random(7)
val_cfgs = [[rng.choice((2, 4, 8)) for _ in range(8)] for _ in range(6)]
val_cfgs += [[4, 4, 4, 4, 2, 4, 8, 8], [8, 8, 4, 4, 4, 4, 4, 4]]
val = [{"name": "val_" + "_".join(map(str, b)), "args": args(b, base_ckpt, E_SENS, 0.02),
        "after": dep} for b in val_cfgs]
b1 = [{"name": f"b1_w{w}a{a}", "args": args([w, a] * 4, fp_init, E_LONG, 0.05)}
      for w, a in PAIRS if (w, a) != (4, 4)]
# interleave so the GPU always has independent work while sensitivity runs wait on the base
order = []
rest = sens + val
while b1 or rest:
    if b1:
        order.append(b1.pop(0))
    for _ in range(4):
        if rest:
            order.append(rest.pop(0))
jobs += order
os.makedirs(out, exist_ok=True)
json.dump(jobs, open(os.path.join(out, "jobs.json"), "w"), indent=1)
print(len(jobs), "jobs;", "validation configs:", val_cfgs)
