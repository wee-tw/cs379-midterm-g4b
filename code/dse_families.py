"""Does the estimator's harm to the DSE depend on the precision family? (estimates)

dse_harm.py over subsets of the bit configurations: every design whose eight widths come
from the allowed set (e.g. {2, 4}: no 8-bit operand anywhere; {8}: uniform INT8), searched
with the biased (FINN) costs and checked against the reference costs. The DSP gap needs an
RTL MVU with an operand wider than 4 bits and PE >= 3, so families without 8-bit widths
should only see the LUT gap. Also reports, for the misjudged designs, the throughput level
(II) and DSP use they would have if the estimate were right, to show why they sit at the
top of the ranking.

  python dse_families.py --truth cost/points_reference.npz --biased cost/points_finn.npz \
      --acc cost/acc_model_3rep.json --out dse_families.json
"""

import argparse
import itertools
import json

import numpy as np
from scipy.stats import kendalltau

from dse_harm import adrs, load, pareto
from energy_search import ALL_CONFIGS, MODES, POWER

FAMILIES = {"all (2/4/8)": (2, 4, 8), "{4, 8}": (4, 8), "{2, 4}": (2, 4), "{2, 8}": (2, 8),
            "INT8 only": (8,), "INT4 only": (4,)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--biased", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = POWER[args.power]
    model = json.load(open(args.acc))
    acc_all = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    ref_acc = model["reference_acc"]
    t_rows, t_e, t_keys = load(args.truth, p)
    b_rows, b_e, b_keys = load(args.biased, p)
    truth = dict(zip(t_keys, t_e))
    bad_all = np.array([k not in truth for k in b_keys])
    out = {}
    # why the misjudged designs rank high: their throughput level and estimated DSPs
    ii_bad, ii_ok = b_rows[bad_all, 3], b_rows[~bad_all, 3]
    out["misjudged_profile"] = {
        "n": int(bad_all.sum()),
        "median_II_misjudged": float(np.median(ii_bad)), "median_II_others": float(np.median(ii_ok)),
        "share_at_fastest_two_II_levels": round(float(np.mean(ii_bad <= np.sort(np.unique(b_rows[:, 3]))[1])), 3),
        "median_DSP_estimate_misjudged": float(np.median(b_rows[bad_all, 6])),
        "modes": {MODES[m]: int(np.sum(b_rows[bad_all, 1] == m)) for m in np.unique(b_rows[bad_all, 1].astype(int))}}
    print("misjudged designs:", out["misjudged_profile"])
    for name, allowed in FAMILIES.items():
        fam = np.array([all(x in allowed for x in b) for b in ALL_CONFIGS])
        bsel = np.where(fam[b_rows[:, 0].astype(int)])[0]
        tsel = np.where(fam[t_rows[:, 0].astype(int)])[0]
        if len(bsel) == 0:
            continue
        bad = np.array([b_keys[i] not in truth for i in bsel])
        res = {"configs": int(fam.sum()), "biased_points": int(len(bsel)),
               "misjudged_pct": round(100 * float(bad.mean()), 2), "margins": {}}
        order_all = bsel[np.argsort(b_e[bsel])]
        res["misjudged_top10_pct"] = round(100 * float(np.mean([b_keys[i] not in truth for i in order_all[:10]])), 1)
        res["misjudged_top100_pct"] = round(100 * float(np.mean([b_keys[i] not in truth for i in order_all[:100]])), 1)
        common = [(b_e[i], truth[b_keys[i]]) for i in bsel if b_keys[i] in truth]
        if len(common) > 2:
            ce = np.array(common)
            res["kendall_tau"] = round(float(kendalltau(ce[:, 0], ce[:, 1])[0]), 4)
        b_acc = acc_all[b_rows[bsel, 0].astype(int)]
        t_acc = acc_all[t_rows[tsel, 0].astype(int)]
        for m in args.margins:
            thr = ref_acc - m
            bi = bsel[b_acc >= thr]
            ti = tsel[t_acc >= thr]
            if len(bi) == 0 or len(ti) == 0:
                continue
            order = bi[np.argsort(b_e[bi])]
            t_opt = ti[np.argmin(t_e[ti])]
            rejects = 0
            chosen = None
            for i in order:
                if b_keys[i] in truth:
                    chosen = i
                    break
                rejects += 1
            r = {"biased_opt_feasible": b_keys[order[0]] in truth, "failed_builds": rejects}
            if chosen is not None:
                r["regret_pct"] = round(100 * (truth[b_keys[chosen]] / t_e[t_opt] - 1), 2)
                r["surprise_pct"] = round(100 * (truth[b_keys[chosen]] / b_e[order[0]] - 1), 2)
            res["margins"][str(m)] = r
        b_front = pareto([(b_e[i], 100 - acc_all[b_keys[i][0]], b_keys[i]) for i in bsel])
        t_front = pareto([(t_e[i], 100 - acc_all[t_keys[i][0]], t_keys[i]) for i in tsel])
        approx = [(truth[k], err, k) for _, err, k in b_front if k in truth]
        res["front"] = {"biased": len(b_front), "infeasible": len(b_front) - len(approx),
                        "ADRS_pct": round(100 * adrs(t_front, approx), 2)}
        out[name] = res
        ms = "; ".join(f"m{m}: {'ok' if r['biased_opt_feasible'] else 'INFEASIBLE'} "
                       f"fails {r['failed_builds']} regret {r.get('regret_pct')}%"
                       for m, r in res["margins"].items())
        print(f"{name:12s} {res['configs']:5d} configs {res['biased_points']:6d} pts | misjudged "
              f"{res['misjudged_pct']}% (top10 {res['misjudged_top10_pct']}%, top100 "
              f"{res['misjudged_top100_pct']}%) tau {res.get('kendall_tau')} | front "
              f"{res['front']['infeasible']}/{res['front']['biased']} infeasible, ADRS "
              f"{res['front']['ADRS_pct']}% | {ms}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
