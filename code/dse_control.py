"""Is it the size of the estimator error or where it sits? Controls for dse_harm.py. (estimates)

The FINN estimate misjudges 0.93% of the design points it calls feasible and ranks designs
with Kendall tau = 0.98 against the reference, yet it invalidates 4 of 5 optima. Two
controls with the same error budget but no structure, on the reference design points:
  A  same misjudgment rate: a random 0.93% of the reference designs are declared infeasible
     on the device while the search still sees them (the search uses reference energies)
  B  same rank agreement: no feasibility errors, but every energy is multiplied by
     exp(sigma z), z ~ N(0, 1) i.i.d., with sigma set so that Kendall's tau with the
     reference is the observed 0.98
Each control is repeated over seeds; reported per margin as in dse_harm.py (optimum
infeasible, failed builds of a synthesize-and-reject loop, regret), plus the ADRS of the
Pareto front.

  python dse_control.py --truth cost/points_reference.npz --acc cost/acc_model_3rep.json \
      --rate 0.0093 --tau 0.983 --seeds 200 --out dse_control.json
"""

import argparse
import json

import numpy as np
from scipy.stats import kendalltau

from dse_harm import adrs, load, pareto
from energy_search import ALL_CONFIGS, POWER


def search(e_search, feasible, acc, thr, e_true):
    """Optimum under e_search; then reject designs that are not feasible on the device."""
    idx = np.where(acc >= thr)[0]
    order = idx[np.argsort(e_search[idx])]
    first_ok = np.argmax(feasible[order])            # index of the first feasible design
    chosen = order[first_ok]
    t = idx[feasible[idx]]
    t_opt = t[np.argmin(e_true[t])]
    return (not feasible[order[0]], int(first_ok),
            100 * (e_true[chosen] / e_true[t_opt] - 1))


def front_adrs(e_search, feasible, err, e_true, keys):
    b_front = pareto([(e_search[i], err[i], i) for i in range(len(e_search))])
    t_front = pareto([(e_true[i], err[i], i) for i in np.where(feasible)[0]])
    approx = [(e_true[i], er, i) for _, er, i in b_front if feasible[i]]
    return 100 * adrs(t_front, approx), len(b_front) - len(approx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--rate", type=float, default=0.0093)
    ap.add_argument("--tau", type=float, default=0.983)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, e_t, keys = load(args.truth, POWER[args.power])
    model = json.load(open(args.acc))
    acc = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])[rows[:, 0].astype(int)]
    err = 100 - acc
    n = len(e_t)
    rng = np.random.default_rng(0)
    out = {"n_designs": n, "rate": args.rate, "tau_target": args.tau}

    # control A: random misjudgments at the observed rate
    res_a = {str(m): {"infeasible": [], "fails": [], "regret": []} for m in args.margins}
    adrs_a, bad_front_a = [], []
    for s in range(args.seeds):
        feasible = np.ones(n, bool)
        feasible[rng.choice(n, int(round(args.rate * n)), replace=False)] = False
        for m in args.margins:
            inf, fails, reg = search(e_t, feasible, acc, model["reference_acc"] - m, e_t)
            r = res_a[str(m)]
            r["infeasible"].append(inf)
            r["fails"].append(fails)
            r["regret"].append(reg)
        if s < 50:   # the front is the same search every time; 50 seeds are enough for ADRS
            a, b = front_adrs(e_t, feasible, err, e_t, keys)
            adrs_a.append(a)
            bad_front_a.append(b)
    out["A_random_misjudgment"] = {
        m: {"P_optimum_infeasible": round(float(np.mean(r["infeasible"])), 4),
            "mean_failed_builds": round(float(np.mean(r["fails"])), 3),
            "max_failed_builds": int(np.max(r["fails"])),
            "mean_regret_pct": round(float(np.mean(r["regret"])), 3)}
        for m, r in res_a.items()}
    out["A_random_misjudgment"]["ADRS_pct_mean"] = round(float(np.mean(adrs_a)), 3)
    out["A_random_misjudgment"]["front_infeasible_mean"] = round(float(np.mean(bad_front_a)), 2)
    print(f"A (random {100 * args.rate:.2f}% misjudged):", json.dumps(out["A_random_misjudgment"]))

    # control B: unstructured multiplicative noise with the observed rank agreement
    sub = rng.choice(n, min(n, 20000), replace=False)
    lo, hi = 0.0, 2.0
    for _ in range(25):   # bisection on sigma for Kendall's tau on a subsample
        sig = (lo + hi) / 2
        z = np.random.default_rng(1).standard_normal(len(sub))
        tau = kendalltau(e_t[sub] * np.exp(sig * z), e_t[sub])[0]
        lo, hi = (sig, hi) if tau > args.tau else (lo, sig)
    sigma = (lo + hi) / 2
    out["B_sigma"] = round(sigma, 4)
    res_b = {str(m): {"infeasible": [], "fails": [], "regret": []} for m in args.margins}
    adrs_b, taus = [], []
    feasible = np.ones(n, bool)
    for s in range(args.seeds // 4):
        e_b = e_t * np.exp(sigma * rng.standard_normal(n))
        if s < 5:
            taus.append(kendalltau(e_b[sub], e_t[sub])[0])
        for m in args.margins:
            inf, fails, reg = search(e_b, feasible, acc, model["reference_acc"] - m, e_t)
            res_b[str(m)]["regret"].append(reg)
        if s < 20:
            adrs_b.append(front_adrs(e_b, feasible, err, e_t, keys)[0])
    out["B_random_noise"] = {m: {"mean_regret_pct": round(float(np.mean(r["regret"])), 3),
                                 "p90_regret_pct": round(float(np.percentile(r["regret"], 90)), 3)}
                             for m, r in res_b.items()}
    out["B_random_noise"]["tau_mean"] = round(float(np.mean(taus)), 4)
    out["B_random_noise"]["ADRS_pct_mean"] = round(float(np.mean(adrs_b)), 3)
    print(f"B (noise sigma {sigma:.4f}, tau {np.mean(taus):.4f}):", json.dumps(out["B_random_noise"]))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
