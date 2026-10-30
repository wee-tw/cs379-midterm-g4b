"""Energy-aware selection under accuracy-model uncertainty (estimates; accuracies measured).

The additive accuracy model predicts acc(b) = base + sum of per-group deltas. A pick whose
prediction clears the threshold by less than the model error can fail when trained
(implementation_notes.md 12.29: two of nine picks missed by 0.42 and 0.05 points). Here a pick
must clear the threshold by z standard errors:

  pred(b) - z * sigma(b) >= threshold,   sigma(b)^2 = sigma_model^2 + sum_g s_g(b)^2

s_g is the standard error of a group's delta from its 3 replicates (spread / 1.69 / sqrt(3),
the expected range of 3 normal draws being 1.69 sigma); sigma_model is the residual spread of
the measured configurations. Reports, per margin, the pick, its energy cost relative to z = 0,
and the measured outcome where the pick was trained.

  python robust_select.py --points cost/points_thr_hi.npz --acc cost/acc_model_3rep.json
"""

import argparse
import json
import math

import numpy as np

from energy_search import ALL_CONFIGS, F_SET, POWER, energies

# measured top-1 of trained picks (exp8, exp9; W4A4 -> 30 ep KD, seed 0) and the validation
# residuals of the 3-replicate model (acc_model_3rep.json)
MEASURED = {"44444448": 84.80, "44224448": 83.28, "24244448": 81.95, "44222248": 81.87,
            "24222448": 80.32, "44222244": 80.94, "22222448": 79.02, "84848484": 84.48,
            "44444488": 85.27, "48484848": 85.62, "44242448": 83.72, "84242448": 84.50,
            "84242248": 83.19, "42442488": 83.44}


def key(b):
    return ",".join(map(str, b))


def group_terms(b):
    w, a = b[0::2], b[1::2]
    return [f"g{g + 1}_w{w[g]}a{a[g]}" for g in range(4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--acc", required=True)
    p.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    p.add_argument("--z", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.645])
    p.add_argument("--power", default="nominal")
    args = p.parse_args()
    m = json.load(open(args.acc))
    acc, spread, ref = m["acc"], m["delta_spread"], m["reference_acc"]
    pred = np.array([acc[key(b)] for b in ALL_CONFIGS])
    resid = [MEASURED[c] - acc[",".join(c)] for c in MEASURED] + m["validation"]["errors"]
    s_model = float(np.std(resid, ddof=1))
    se = np.array([math.sqrt(s_model ** 2 + sum((spread.get(t, 0.0) / 1.69 / math.sqrt(3)) ** 2
                                                 for t in group_terms(b))) for b in ALL_CONFIGS])
    print(f"residuals of {len(resid)} measured configs: mean {np.mean(resid):+.2f}, sd {s_model:.2f}; "
          f"sigma(b) median {np.median(se):.2f}, range {se.min():.2f}-{se.max():.2f}")
    rows = np.load(args.points)["rows"]
    q = POWER[args.power]
    e = np.min(np.stack([energies(rows, q, f)[1] for f in F_SET]), axis=0)
    best = {}
    for ci, ei in zip(rows[:, 0].astype(int), e):
        best[ci] = min(best.get(ci, np.inf), ei)
    cfgs = np.array(sorted(best))
    eb = np.array([best[c] for c in cfgs])
    for mg in args.margins:
        thr = ref - mg
        base = None
        line = []
        for z in args.z:
            ok = pred[cfgs] - z * se[cfgs] >= thr
            if not ok.any():
                line.append(f"z={z:g}: none")
                continue
            i = int(np.argmin(np.where(ok, eb, np.inf)))
            c = "".join(map(str, ALL_CONFIGS[cfgs[i]]))
            base = base or eb[i]
            meas = MEASURED.get(c)
            tag = "" if meas is None else (" pass" if meas >= thr else f" FAIL {meas - thr:+.2f}")
            line.append(f"z={z:g}: {c} {1e3 * eb[i]:.4f} mJ ({100 * (eb[i] / base - 1):+.1f}%)"
                        f" pred {pred[cfgs[i]]:.2f}{tag}")
        print(f"margin {mg:g} (thr {thr:.2f}): " + " | ".join(line))


if __name__ == "__main__":
    main()
