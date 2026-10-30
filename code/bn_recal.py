"""BatchNorm re-estimation for trained QAT checkpoints (diagnostic, measured accuracy).

Late in low-bit QAT the latent weights keep flipping between quantisation levels, so the BN
running statistics, an average over many weight versions, may not match the final weights
(Nagel et al., ICML 2022). This re-estimates every BN layer's mean and variance with the
final weights on the training images (no labels, no test data, no weight update) and
reports test top-1 before and after. Every run JSON written by train_cifar.py names its
bits, recipe and checkpoint.

  python bn_recal.py --data cifar10.npz --runs exp7/*.json --out bn_recal.json
"""

import argparse
import glob
import json
import os

import torch

from train_cifar import build, evaluate, load_cifar, to_float


def recal(model, x, bs=500):
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches
    model.eval()  # quantisers stay in eval mode, only BN collects statistics
    for m in bns:
        m.train()
    with torch.no_grad():
        for k in range(0, len(x), bs):
            model(to_float(x[k:k + bs]))
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="run JSONs (globs allowed)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, _, xte, yte = load_cifar(args.data, dev)
    files = sorted({f for r in args.runs for f in glob.glob(r)})
    rows = []
    for f in files:
        run = json.load(open(f))
        if not isinstance(run, dict) or "args" not in run:   # summaries, not run records
            continue
        a = run["args"]
        ckpt = os.path.splitext(f)[0] + ".pt"
        if a.get("model") != "brevitas" or not os.path.exists(ckpt):
            continue
        model = build(a["model"], a["bits"], a["recipe"], a.get("amid"))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))  # as train_cifar.py
        model = model.to(dev)
        before = evaluate(model, xte, yte)
        recal(model, xtr)
        after = evaluate(model, xte, yte)
        name = os.path.basename(os.path.dirname(f)) + "/" + os.path.splitext(os.path.basename(f))[0]
        rows.append({"run": name, "bits": "".join(map(str, a["bits"])), "init": a.get("init"),
                     "epochs": a["epochs"], "logged": run["final_acc"], "before": before,
                     "after": after})
        print(f"{name:34s} {rows[-1]['bits']}  logged {run['final_acc']:6.2f}  eval {before:6.2f}"
              f"  BN re-estimated {after:6.2f}  ({after - before:+.2f})", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
