"""Feasible@K: share of an estimator's K most energy-efficient designs (within the accuracy
constraint) that the reference finds feasible, the hardware analogue of P@topK (12.32; estimates).

  python feasible_at_k.py --truth cost/points_reference_calv2.npz --biased finn=cost/points_finn_calv2.npz \
      --acc cost/acc_model_3rep.json
"""
import argparse
import json

import numpy as np

from dse_harm import load
from energy_search import ALL_CONFIGS, POWER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--biased", nargs="+", required=True, help="name=points.npz")
    ap.add_argument("--acc", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--power", default="nominal")
    args = ap.parse_args()
    p = POWER[args.power]
    m = json.load(open(args.acc))
    acc = np.array([m["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    t_rows, t_e, t_keys = load(args.truth, p)
    truth = dict(zip(t_keys, t_e))
    print("| model | margin | " + " | ".join(f"Feasible@{k}" for k in args.k) + " | P@top10 |")
    print("|---|---|" + "---|" * (len(args.k) + 1))
    for spec in args.biased:
        name, path = spec.split("=", 1)
        b_rows, b_e, b_keys = load(path, p)
        for mg in args.margins:
            thr = m["reference_acc"] - mg
            ok_b = acc[b_rows[:, 0].astype(int)] >= thr
            order = [b_keys[i] for i in np.argsort(b_e) if ok_b[i]]
            ok_t = acc[t_rows[:, 0].astype(int)] >= thr
            t_top10 = set([t_keys[i] for i in np.argsort(t_e) if ok_t[i]][:10])
            fk = [100.0 * np.mean([k in truth for k in order[:K]]) for K in args.k]
            p10 = len(t_top10 & set(order[:10])) / 10
            print(f"| {name} | {mg:g} | " + " | ".join(f"{x:.0f}%" for x in fk) + f" | {p10:.1f} |")


if __name__ == "__main__":
    main()
