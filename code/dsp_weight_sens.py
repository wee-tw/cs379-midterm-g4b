"""How much does the energy-optimal mapping depend on the assumed power of a DSP slice?

U = LUT fraction + w_BRAM * BRAM fraction + w_DSP * DSP fraction, with w = 0.25 an
assumption (implementation_notes.md 12.16). Once DSP-based RTL MVUs are costed correctly
(cost_tables.py modes dspc / dsp4c), A' can move whole layers from LUTs into DSPs, so
its energy and its mapping depend on w_DSP. This reruns the search at one accuracy margin
for several w_DSP (BRAM weight unchanged) and reports A' and the baselines (estimates).

  python dsp_weight_sens.py --points cost/points_corr_lo.npz --acc cost/acc_model_3rep.json
"""

import argparse
import json

import numpy as np

import energy_search as es


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--acc", required=True)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = np.load(args.points)["rows"]
    model = json.load(open(args.acc))
    acc = [model["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS]
    thr = model["reference_acc"] - args.margin
    w0 = es.UTIL_W["DSP"]
    out = {}
    for w in args.weights:
        es.UTIL_W["DSP"] = w
        for pname in ("nominal", "z2_anchor"):
            sel = es.search(rows, acc, thr, es.POWER[pname])["continuous"]
            a = sel["A_prime"]
            gaps = {k: round(100 * (v["E_continuous_mJ"] / a["E_continuous_mJ"] - 1), 1)
                    for k, v in sel.items() if k != "A_prime"}
            out.setdefault(str(w), {})[pname] = {"A_prime": a, "gaps_pct": gaps}
            print(f"w_DSP {w:<4} {pname:9s} A' {''.join(map(str, a['bits']))} {a['mode']:6s} "
                  f"LUT {a['LUT']:6d} DSP {a['DSP']:3d} U {a['U']:.3f} E {a['E_continuous_mJ']:.4f} mJ | "
                  + ", ".join(f"{k.split('_')[0]} {v:+.1f}%" for k, v in gaps.items()))
    es.UTIL_W["DSP"] = w0
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
