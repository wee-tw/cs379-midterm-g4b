"""Every QAT run of a named configuration, grouped by training protocol, next to the
accuracy model's prediction. The additive model is fitted to 8-epoch fine-tunes from the
uniform W4A4 model (sens_*), so runs with that protocol are out-of-sample checks and the
other protocols show how far the prediction carries.

  python cand_table.py --acc 1rep=cost/acc_model.json 2rep=cost/acc_model_2rep.json \
      --dirs exp3 exp4 exp5 exp6 exp7 [--out cand_table.md]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

SKIP = ("sens_", "rs_", "abl_", "finnfloat", "jobs")


def protocol(a):
    init = os.path.basename(str(a.get("init")))
    src = "float" if init.startswith("finnfloat") else "W4A4" if init.startswith("sens_base") else init
    kd = a.get("teacher") is not None and float(a.get("kd_alpha", 0) or 0) > 0
    return f"{src}→{a['epochs']} ep{' KD' if kd else ''} lr {a['lr']:g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acc", nargs="+", default=[], help="label=acc_model.json")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out")
    p.add_argument("--json", help="also write the per-protocol errors (for make_figures.py --protocol)")
    args = p.parse_args()
    models = [(s.split("=", 1)[0], json.load(open(s.split("=", 1)[1]))) for s in args.acc]
    runs = defaultdict(list)
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.basename(f)[:-5]
            if name.startswith(SKIP):
                continue
            r = json.load(open(f))
            a = r.get("args")
            if not isinstance(a, dict) or a.get("recipe") != "lowbit" or "bits" not in a:
                continue
            key = ("".join(map(str, a["bits"])), protocol(a))
            runs[key].append({"acc": r["final_acc"], "seed": a.get("seed"),
                              "run": f"{os.path.basename(d)}/{name}"})
    head = "| config | protocol | n | top-1 mean ± sd (%) | " + " | ".join(
        f"pred {lab}" for lab, _ in models) + " | runs |"
    lines = [head, "|" + "---|" * (5 + len(models))]
    err = defaultdict(lambda: defaultdict(list))
    by_w2 = defaultdict(list)        # (protocol, model, number of 2-bit-weight groups)
    for (cfg, proto), rs in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        accs = [x["acc"] for x in rs]
        m, sd = float(np.mean(accs)), (float(np.std(accs, ddof=1)) if len(accs) > 1 else None)
        preds = []
        for lab, am in models:
            pr = am["acc"].get(",".join(cfg))
            preds.append(pr)
            if pr is not None:
                err[proto][lab].append(m - pr)
                by_w2[(proto, lab, cfg[0::2].count("2"))].append(m - pr)
        lines.append(f"| {cfg} | {proto} | {len(accs)} | {m:.2f}"
                     + (f" ± {sd:.2f}" if sd is not None else "") + " | "
                     + " | ".join("—" if x is None else f"{x:.2f}" for x in preds) + " | "
                     + ", ".join(f"{x['run']} (s{x['seed']}) {x['acc']}" for x in rs) + " |")
    lines += ["", "| protocol | model | n configs | mean (measured − predicted) | MAE |", "|---|---|---|---|---|"]
    for proto, e in sorted(err.items()):
        for lab, v in e.items():
            lines.append(f"| {proto} | {lab} | {len(v)} | {np.mean(v):+.2f} | {np.mean(np.abs(v)):.2f} |")
    lines += ["", "| protocol | model | 2-bit-weight groups | n configs | mean (measured − predicted) "
              "| worst |", "|---|---|---|---|---|---|"]
    for (proto, lab, k), v in sorted(by_w2.items()):
        lines.append(f"| {proto} | {lab} | {k} | {len(v)} | {np.mean(v):+.2f} | {min(v):+.2f} |")
    txt = "\n".join(lines) + "\n"
    if args.json:
        json.dump({"by_w2": [{"protocol": pr, "model": lab, "w2_groups": k, "errors": v}
                             for (pr, lab, k), v in sorted(by_w2.items())],
                   "runs": [{"config": c, "protocol": pr, "acc": [x["acc"] for x in rs]}
                            for (c, pr), rs in sorted(runs.items())]},
                  open(args.json, "w"), indent=1)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
