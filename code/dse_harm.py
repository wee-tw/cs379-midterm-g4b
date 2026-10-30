"""How much does a resource-estimator error hurt the design-space exploration? (estimates)

The same design points are costed by a biased model (what the designer searches with) and
by a reference model (what the hardware does). Both come from dse_variants.py, so a point
(bits, mode, T) is the same folding in both. For each accuracy margin (additive accuracy
model, threshold = reference uniform prediction - margin) it reports:
  optimum      the biased optimum (minimum continuous energy over F_SET) and whether the
               reference finds it feasible
  build loop   synthesize-and-reject: walk the biased ranking and reject every design the
               reference finds infeasible (one failed build each) until one fits; regret of
               that design against the reference optimum, and its "postdecision surprise"
               (reference energy over the energy the biased model promised for the
               optimum; Smith & Winkler 2006)
  top-k        overlap of the k best designs under each model
  ADRS         average distance from the reference Pareto front (energy vs predicted top-1
               error) to the biased front re-costed with the reference (Palermo et al.)
and over all designs feasible under both: Kendall's tau of the two energies, which is the
"preserves the ranking" criterion estimators are usually validated with.

  python dse_harm.py --truth cost/points_reference.npz --biased finn=cost/points_finn.npz \
      lut_fixed=cost/points_lut_fixed.npz --acc cost/acc_model_3rep.json --out dse_harm.json
"""

import argparse
import json

import numpy as np
from scipy.stats import kendalltau

from energy_search import ALL_CONFIGS, F_SET, MODES, POWER, energies


def load(path, p):
    rows = np.load(path)["rows"]
    e = np.min(np.stack([energies(rows, p, f)[1] for f in F_SET]), axis=0)
    keys = [(int(r[0]), int(r[1]), int(r[2])) for r in rows]
    return rows, e, keys


def pareto(points):
    """points: list of (energy, error, key); both minimised. Returns the non-dominated set."""
    front, best = [], np.inf
    for e, err, k in sorted(points, key=lambda t: (t[1], t[0])):
        if e < best:
            front.append((e, err, k))
            best = e
    return front


def adrs(ref_front, approx):
    if not approx:
        return float("nan")
    d = []
    for e_r, err_r, _ in ref_front:
        d.append(min(max(0.0, (e_a - e_r) / e_r, (err_a - err_r) / err_r) for e_a, err_a, _ in approx))
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--biased", nargs="+", required=True, help="name=points.npz")
    ap.add_argument("--acc", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--power", nargs="+", default=["nominal", "z2_anchor"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = json.load(open(args.acc))
    acc = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    ref_acc = model["reference_acc"]
    out = {}
    for pname in args.power:
        p = POWER[pname]
        t_rows, t_e, t_keys = load(args.truth, p)
        truth = dict(zip(t_keys, t_e))
        t_acc = acc[t_rows[:, 0].astype(int)]
        t_front = pareto([(e, 100 - a, k) for e, a, k in zip(t_e, t_acc, t_keys)])
        for spec in args.biased:
            name, path = spec.split("=")
            b_rows, b_e, b_keys = load(path, p)
            b_acc = acc[b_rows[:, 0].astype(int)]
            res = {"margins": {}}
            # global ranking agreement over designs feasible under both models
            common = [(e, truth[k], k[1]) for e, k in zip(b_e, b_keys) if k in truth]
            ce = np.array([(a, b) for a, b, _ in common])
            dsp_rows = np.array([(a, b) for a, b, m in common if MODES[m].startswith("dsp")])
            res["kendall_tau_all"] = round(float(kendalltau(ce[:, 0], ce[:, 1])[0]), 4)
            res["kendall_tau_dsp_designs"] = (round(float(kendalltau(dsp_rows[:, 0], dsp_rows[:, 1])[0]), 4)
                                              if len(dsp_rows) > 2 else None)
            infeas = [k for k in b_keys if k not in truth]
            res["biased_feasible_points"] = len(b_keys)
            res["of_which_infeasible_in_truth"] = len(infeas)
            for m in args.margins:
                thr = ref_acc - m
                bi = np.where(b_acc >= thr)[0]
                order = bi[np.argsort(b_e[bi])]
                ti = np.where(t_acc >= thr)[0]
                t_opt = ti[np.argmin(t_e[ti])]
                rejects, chosen = 0, None
                for i in order:
                    if b_keys[i] in truth:
                        chosen = i
                        break
                    rejects += 1
                r = {"threshold": round(thr, 2),
                     "biased_opt": {"bits": "".join(map(str, ALL_CONFIGS[b_keys[order[0]][0]])),
                                    "mode": MODES[b_keys[order[0]][1]], "T": b_keys[order[0]][2],
                                    "E_promised_mJ": round(float(b_e[order[0]]) * 1e3, 4),
                                    "feasible_in_truth": b_keys[order[0]] in truth},
                     "truth_opt": {"bits": "".join(map(str, ALL_CONFIGS[t_keys[t_opt][0]])),
                                   "mode": MODES[t_keys[t_opt][1]],
                                   "E_mJ": round(float(t_e[t_opt]) * 1e3, 4)},
                     "failed_builds": rejects}
                if chosen is not None:
                    e_true = truth[b_keys[chosen]]
                    r["after_rejects"] = {"bits": "".join(map(str, ALL_CONFIGS[b_keys[chosen][0]])),
                                          "mode": MODES[b_keys[chosen][1]],
                                          "E_true_mJ": round(float(e_true) * 1e3, 4),
                                          "regret_pct": round(100 * (e_true / t_e[t_opt] - 1), 2),
                                          "surprise_pct": round(100 * (e_true / b_e[order[0]] - 1), 2)}
                for k in (1, 10, 100):
                    top_b = {b_keys[i] for i in order[:k]}
                    top_t = {t_keys[i] for i in ti[np.argsort(t_e[ti])][:k]}
                    r[f"top{k}_overlap"] = round(len(top_b & top_t) / k, 3)
                res["margins"][str(m)] = r
            b_front = pareto([(e, 100 - a, k) for e, a, k in zip(b_e, b_acc, b_keys)])
            approx = [(truth[k], err, k) for e, err, k in b_front if k in truth]
            res["pareto"] = {"truth_front": len(t_front), "biased_front": len(b_front),
                             "biased_front_infeasible": len(b_front) - len(approx),
                             "ADRS_pct": round(100 * adrs(t_front, approx), 2)}
            out.setdefault(pname, {})[name] = res
            print(f"== {pname} | biased model '{name}': tau(all) {res['kendall_tau_all']}, "
                  f"tau(dsp designs) {res['kendall_tau_dsp_designs']}; {len(infeas)} of {len(b_keys)} "
                  f"feasible-looking points infeasible; Pareto front {len(b_front)} designs, "
                  f"{res['pareto']['biased_front_infeasible']} infeasible, ADRS {res['pareto']['ADRS_pct']}%")
            for m, r in res["margins"].items():
                a = r.get("after_rejects", {})
                print(f"   margin {m}: biased opt {r['biased_opt']['bits']} {r['biased_opt']['mode']} "
                      f"{'ok' if r['biased_opt']['feasible_in_truth'] else 'INFEASIBLE'}; "
                      f"failed builds {r['failed_builds']}; then {a.get('bits')} {a.get('mode')} "
                      f"regret {a.get('regret_pct')}% surprise {a.get('surprise_pct')}% | "
                      f"truth opt {r['truth_opt']['bits']} {r['truth_opt']['mode']} | "
                      f"top1/10/100 {r['top1_overlap']}/{r['top10_overlap']}/{r['top100_overlap']}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
