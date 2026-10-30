"""Remedies for a biased resource estimator in DSE, compared on equal synthesis budgets (12.32; estimates).

The harm study (dse_harm.py) costs the same designs with FINN's estimates (biased) and with
the calibrated reference (truth). Here a "synthesis" of a design reveals its truth: whether it
fits and its true energy. Strategies that start from the biased estimator, after the
literature on DSE with uncertain feasibility:
  reject     walk the biased ranking and synthesize until a design fits (12.25's build loop)
  topK       synthesize the biased top K at once and keep the best that fits (GNN-DSE: top 10)
  margin     derate the DSP budget by d before ranking, then reject (12.26's safety margin)
  learned    HyperMapper-style feasibility learning: a random forest on design features,
             initialized with n0 random syntheses, then active learning (synthesize the
             cheapest design predicted feasible, retrain), budget B; 10 seeds
  calibrated the calibrated estimator itself, against a truth with residual design-level
             LUT noise (sigma 5%), 10 seeds: the remedy we propose
Per margin: syntheses until the first design that fits, regret of the final pick against the
true optimum, and whether the optimum was found. `mechanism` fits a shallow decision tree to
the designs the biased model misjudges and tabulates the misjudgment rate by DSP slack.

  python dse_remedies.py --truth cost/points_reference_cal.npz --biased cost/points_finn_cal.npz \
      --acc cost/acc_model_3rep.json --out cost/dse_remedies.json
"""

import argparse
import json

import numpy as np

from assemble_costs import DEVICE
from energy_search import ALL_CONFIGS, DERATE, F_SET, MODES, POWER, energies

BUDGET = {"LUT": DEVICE["LUT"] * DERATE["LUT"], "BRAM": DEVICE["BRAM_18K"] * DERATE["BRAM_18K"],
          "DSP": DEVICE["DSP"] * DERATE["DSP"]}


def load(path, p):
    rows = np.load(path)["rows"]
    e = np.min(np.stack([energies(rows, p, f)[1] for f in F_SET]), axis=0)
    return rows, e


def keys_of(rows):
    return [(int(r[0]), int(r[1]), int(r[2])) for r in rows]


def features(rows):
    bits = np.array([ALL_CONFIGS[int(c)] for c in rows[:, 0]], dtype=float)
    mode = np.eye(len(MODES))[rows[:, 1].astype(int)]
    return np.hstack([bits, mode, np.log(rows[:, 2:4]), rows[:, 4:7]])


def strategies(b_rows, b_e, ok, true_e, cand, e_star, seeds, n0, budgets):
    order = cand[np.argsort(b_e[cand])]
    res = {}
    # reject: sequential synthesize-and-reject along the biased ranking
    first = next((i for i, j in enumerate(order) if ok[j]), None)
    res["reject"] = {"syntheses": None if first is None else first + 1,
                     "regret_pct": None if first is None else 100 * (true_e[order[first]] / e_star - 1)}
    for K in (10, 30):
        top = [j for j in order[:K] if ok[j]]
        best = min((true_e[j] for j in top), default=None)
        res[f"top{K}"] = {"syntheses": K, "found_feasible": bool(top),
                          "regret_pct": None if best is None else 100 * (best / e_star - 1)}
    for d in (0.2, 0.35, 0.5):
        o = order[b_rows[order, 6] <= BUDGET["DSP"] * (1 - d)]
        first = next((i for i, j in enumerate(o) if ok[j]), None)
        res[f"margin{int(100 * d)}"] = {"syntheses": None if first is None else first + 1,
                                        "regret_pct": None if first is None else 100 * (true_e[o[first]] / e_star - 1)}
    # learned feasibility with active learning
    from sklearn.ensemble import RandomForestClassifier
    X = features(b_rows)
    pool = order[:2000]  # the learner only ever picks among the cheapest designs
    runs = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        labeled = list(rng.choice(cand, size=min(n0, len(cand)), replace=False))
        hist = []
        for _ in range(max(budgets) - len(labeled)):
            y = ok[labeled]
            if y.all() or not y.any():
                p = np.full(len(pool), y.mean())
            else:
                clf = RandomForestClassifier(n_estimators=100, random_state=s, class_weight="balanced")
                clf.fit(X[labeled], y)
                p = clf.predict_proba(X[pool])[:, 1]
            seen = set(labeled)
            masked = np.where(p >= 0.5, b_e[pool], np.inf)
            pick = next((pool[i] for i in np.argsort(masked) if np.isfinite(masked[i]) and pool[i] not in seen),
                        None)
            if pick is None:
                break
            labeled.append(pick)
            feas = [true_e[j] for j in labeled if ok[j]]
            hist.append(min(feas) if feas else np.inf)
        feas_all = [true_e[j] for j in labeled if ok[j]]
        runs.append({"by_budget": {B: (100 * (min([true_e[j] for j in labeled[:B] if ok[j]], default=np.inf)
                                               / e_star - 1)) for B in budgets},
                     "found_within_1pct_at": next((n0 + i + 1 for i, v in enumerate(hist)
                                                   if v <= 1.01 * e_star), None)})
    hits = [r["found_within_1pct_at"] for r in runs]
    res["learned"] = {"n0": n0, "seeds": seeds,
                      "syntheses_to_1pct": {"median": float(np.median([h if h else np.inf for h in hits])),
                                            "max": max(h if h else np.inf for h in hits)},
                      "median_regret_pct_by_budget": {B: float(np.median([r["by_budget"][B] for r in runs]))
                                                     for B in budgets},
                      "share_found_within_1pct": {B: float(np.mean([(r["found_within_1pct_at"] or 1e9) <= B
                                                                   for r in runs])) for B in budgets}}
    return res


def calibrated_with_noise(t_rows, t_e, t_acc_ok, seeds, sigma):
    """The calibrated model ranks; the truth differs from it by design-level LUT noise."""
    out = []
    cand = np.where(t_acc_ok)[0]
    order = cand[np.argsort(t_e[cand])]
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        lut = t_rows[:, 4] * (1 + sigma * rng.standard_normal(len(t_rows)))
        fits = (lut <= BUDGET["LUT"]) & (t_rows[:, 5] <= BUDGET["BRAM"]) & (t_rows[:, 6] <= BUDGET["DSP"])
        e_star = t_e[cand][fits[cand]].min()
        first = next(i for i, j in enumerate(order) if fits[j])
        out.append((first + 1, 100 * (t_e[order[first]] / e_star - 1)))
    return {"sigma": sigma, "seeds": seeds, "median_syntheses": float(np.median([o[0] for o in out])),
            "max_syntheses": int(max(o[0] for o in out)), "median_regret_pct": float(np.median([o[1] for o in out]))}


def mechanism(b_rows, ok):
    from sklearn.tree import DecisionTreeClassifier, export_text
    X = features(b_rows)
    names = [f"bit{i}" for i in range(8)] + [f"mode_{m}" for m in MODES] + ["logT", "logcyc", "LUT_est", "BRAM_est", "DSP_est"]
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(X, ~ok)
    slack = BUDGET["DSP"] - b_rows[:, 6]
    bins = [(0, 20), (20, 50), (50, 100), (100, 1e9)]
    by_slack = {f"{lo}-{hi if hi < 1e9 else 'inf'}": {"n": int(((slack >= lo) & (slack < hi)).sum()),
                                                     "misjudged_pct": float(100 * (~ok)[(slack >= lo) & (slack < hi)].mean())
                                                     if ((slack >= lo) & (slack < hi)).any() else None}
                for lo, hi in bins}
    by_mode = {MODES[int(m)]: float(100 * (~ok)[b_rows[:, 1] == m].mean()) for m in np.unique(b_rows[:, 1])}
    return {"tree": export_text(tree, feature_names=names, decimals=0), "by_dsp_slack": by_slack, "by_mode": by_mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--biased", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = POWER["nominal"]
    t_rows, t_e = load(args.truth, p)
    b_rows, b_e = load(args.biased, p)
    truth = dict(zip(keys_of(t_rows), t_e))
    bk = keys_of(b_rows)
    ok = np.array([k in truth for k in bk])
    true_e = np.array([truth.get(k, np.inf) for k in bk])
    m = json.load(open(args.acc))
    acc = np.array([m["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    out = {"budget": BUDGET, "margins": {}}
    for mg in args.margins:
        thr = m["reference_acc"] - mg
        cand = np.where(acc[b_rows[:, 0].astype(int)] >= thr)[0]
        t_ok = acc[t_rows[:, 0].astype(int)] >= thr
        e_star = t_e[t_ok].min()
        r = strategies(b_rows, b_e, ok, true_e, cand, e_star, args.seeds, args.n0, args.budgets)
        r["calibrated"] = calibrated_with_noise(t_rows, t_e, t_ok, args.seeds, 0.05)
        out["margins"][str(mg)] = r
        print(f"== margin {mg:g}: true optimum {1e3 * e_star:.4f} mJ")
        for k, v in r.items():
            print(f"   {k:10s} {v}")
    out["mechanism"] = mechanism(b_rows, ok)
    print(out["mechanism"]["tree"])
    print("misjudged by DSP slack:", out["mechanism"]["by_dsp_slack"])
    print("misjudged by mode:", out["mechanism"]["by_mode"])
    json.dump(out, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
