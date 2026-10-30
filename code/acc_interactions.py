"""Pairwise correction of the additive accuracy model where it fails: several groups
with 2-bit weights at once (acc_model.json overestimates uniform W2 by 11-22 points).

    acc'(b) = acc_additive(b) + sum_{i<j} gamma_ij * [W_i = 2] * [W_j = 2]

gamma is fitted by ridge least squares on measured runs that have two or more W2 groups
and the same protocol as the sensitivity scan (8-epoch fine-tunes from the all-W4A4
base): exp5 int_<bits> runs plus exp3 val_<bits> runs. The uniform 30-epoch runs
(b1_w2a*) use a different protocol and are only reported as an out-of-sample check.

  python acc_interactions.py --acc acc_model.json --runs exp3 exp5 --out acc_model_int.json
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

PAIRS = list(itertools.combinations(range(4), 2))


def final_acc(path):
    return json.load(open(path))["history"][-1]["test_acc"]


def w2_pairs(bits):
    w = bits[0::2]
    return [int(w[i] == 2 and w[j] == 2) for i, j in PAIRS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acc", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ridge", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    a = json.load(open(args.acc))
    X, y, names = [], [], []
    for d in args.runs:
        for f in sorted(glob.glob(os.path.join(d, "int_*.json")) +
                        glob.glob(os.path.join(d, "val_*.json"))):
            bits = [int(v) for v in os.path.basename(f)[:-5].split("_")[1:]]
            x = w2_pairs(bits)
            if sum(x) == 0:
                continue
            key = ",".join(map(str, bits))
            X.append(x)
            y.append(final_acc(f) - a["acc"][key])
            names.append(key)
    X, y = np.array(X, float), np.array(y)
    gamma = np.linalg.solve(X.T @ X + args.ridge * np.eye(len(PAIRS)), X.T @ y)
    fit = X @ gamma
    print("runs with >=2 W2 groups:", len(y))
    for n, r, ff in zip(names, y, fit):
        print(f"  {n}: additive residual {r:+.2f}, pairwise fit {ff:+.2f}")
    print("gamma:", {f"g{i + 1}g{j + 1}": round(float(g), 2) for (i, j), g in zip(PAIRS, gamma)})
    acc2 = {k: round(v + float(np.dot(w2_pairs([int(t) for t in k.split(",")]), gamma)), 2)
            for k, v in a["acc"].items()}
    # out-of-sample: uniform W2 (different protocol: 30 epochs from the float model)
    for d in args.runs:
        for f in sorted(glob.glob(os.path.join(d, "b1_w2a*.json"))):
            wa = os.path.basename(f)[3:-5]
            key = ",".join([wa[1], wa[3]] * 4)
            print(f"  out-of-sample {wa}: measured {final_acc(f):.2f}, additive {a['acc'][key]:.2f}, "
                  f"pairwise {acc2[key]:.2f}")
    ref = a["reference_acc"]
    out = dict(a, acc=acc2, threshold=ref - args.margin,
               interactions={"gamma": {f"g{i + 1}g{j + 1}": float(g) for (i, j), g in zip(PAIRS, gamma)},
                             "n_runs": len(y), "ridge": args.ridge})
    json.dump(out, open(args.out, "w"))
    thr = ref - args.margin
    print(f"threshold {thr:.2f}: {sum(v >= thr for v in acc2.values())} configs pass "
          f"(additive: {sum(v >= thr for v in a['acc'].values())})")


if __name__ == "__main__":
    main()
