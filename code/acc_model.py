"""Additive accuracy model for all 6,561 ResNet-8 bit configurations, from a per-block
sensitivity scan (measured QAT runs).

  python acc_model.py --runs exp3 --base sens_base --out acc_model.json --margin 1.0 \
      --reference b1_w4a4 [--check val_*]

Each sensitivity run is named sens_g<g>_w<W>a<A> (group g in 1..4 changed to (W, A), all
other groups at the base pair) and was fine-tuned from the same base checkpoint for the
same number of epochs as the control run `--base`. Then
    acc(b) = acc(control) + sum_g [acc(sens_g, b_g) - acc(control)]
The accuracy threshold is acc(--reference) - margin, i.e. "no more than `margin` points
below the reference (tuned uniform) model", as in the plan.
--check compares the prediction with measured runs named val_<w1>_<a1>_..._<a4>.
"""

import argparse
import glob
import itertools
import json
import os


def final_acc(path):
    # last epoch: with OneCycle it is the only one at the final (near-zero) learning
    # rate; earlier epochs are visibly less converged for 2-bit runs
    return json.load(open(path))["history"][-1]["test_acc"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="one or more run folders, each a full scan with its own control; "
                        "deltas are averaged over them (replicate seeds)")
    p.add_argument("--base", required=True, help="control run name (base config fine-tune)")
    p.add_argument("--base-pair", type=int, nargs=2, default=[4, 4])
    p.add_argument("--reference", required=True, help="run whose accuracy sets the threshold")
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--check", default=None, help="glob of validation runs")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    bases = {d: final_acc(os.path.join(d, args.base + ".json")) for d in args.runs
             if os.path.exists(os.path.join(d, args.base + ".json"))}
    base = sum(bases.values()) / len(bases)
    delta, spread = {}, {}
    for g in range(4):
        for w, a in itertools.product((2, 4, 8), repeat=2):
            if (w, a) == tuple(args.base_pair):
                delta[(g, w, a)] = 0.0
                continue
            ds = [final_acc(f) - bases[d] for d in bases
                  for f in [os.path.join(d, f"sens_g{g + 1}_w{w}a{a}.json")] if os.path.exists(f)]
            delta[(g, w, a)] = sum(ds) / len(ds) if ds else None
            if len(ds) > 1:
                spread[f"g{g + 1}_w{w}a{a}"] = round(max(ds) - min(ds), 2)
    missing = [k for k, v in delta.items() if v is None]
    if missing:
        print("missing sensitivity runs:", missing)
    acc = {}
    for b in itertools.product((2, 4, 8), repeat=8):
        ds = [delta[(g, b[2 * g], b[2 * g + 1])] for g in range(4)]
        if any(d is None for d in ds):
            continue
        acc[",".join(map(str, b))] = round(base + sum(ds), 2)
    if args.reference == "uniform_model":
        # same scale as the model: best predicted uniform configuration
        ref = max(v for k, v in acc.items() if len(set(zip(*[iter(k.split(","))] * 2))) == 1)
    else:
        ref = final_acc(os.path.join(args.runs[0], args.reference + ".json"))
    thr = ref - args.margin
    out = {"base_acc": base, "reference_acc": ref, "threshold": thr,
           "replicates": len(bases), "delta_spread": spread,
           "delta": {f"g{g + 1}_w{w}a{a}": v for (g, w, a), v in delta.items()}, "acc": acc}
    if args.check:
        errs = []
        for f in sorted(glob.glob(os.path.join(args.runs[0], args.check + ".json"))):
            bits = os.path.basename(f)[4:-5].split("_")
            key = ",".join(bits)
            if key in acc:
                meas = final_acc(f)
                errs.append(acc[key] - meas)
                print(f"{key}: predicted {acc[key]:.2f}  measured {meas:.2f}  "
                      f"error {acc[key] - meas:+.2f}")
        if errs:
            mae = sum(abs(e) for e in errs) / len(errs)
            out["validation"] = {"n": len(errs), "mae": mae, "errors": errs}
            print(f"additive model MAE over {len(errs)} configs: {mae:.2f} points")
    json.dump(out, open(args.out, "w"))
    print(f"base {base:.2f}, reference {ref:.2f}, threshold {thr:.2f}, "
          f"{sum(v >= thr for v in acc.values())}/{len(acc)} configs pass")


if __name__ == "__main__":
    main()
