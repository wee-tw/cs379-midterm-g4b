"""H3 (preliminary, model-based): do BOPs / weight memory rank configurations the way
FINN's hardware cost and the energy model do?

For every bit configuration it takes, from the feasible design points:
  * min LUT needed to reach a throughput target (FINN estimate, best mode/folding)
  * min continuous energy and min fixed-rate energy (energy model, best mode/folding/f)
and reports Kendall's tau / Spearman's rho against BOPs and weight-memory bits, plus how
often the proxy-best configuration (lowest BOPs) is also the cost-best one.

  python analysis_proxy.py --points points.npz --out proxy.json [--acc acc_model.json]
"""

import argparse
import json

import numpy as np
from scipy.stats import kendalltau, spearmanr

from energy_search import ALL_CONFIGS, F_SET, POWER, bops, energies, wmem_bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--acc", default=None)
    p.add_argument("--fps", type=float, default=2000.0, help="throughput target for min-LUT")
    p.add_argument("--rate", type=float, default=300.0, help="fixed rate R for E_r")
    args = p.parse_args()
    rows = np.load(args.points)["rows"]
    cfg = rows[:, 0].astype(int)
    n = len(ALL_CONFIGS)
    keep = np.ones(n, bool)
    if args.acc:
        a = json.load(open(args.acc))
        keep = np.array([a["acc"][",".join(map(str, b))] >= a["threshold"] for b in ALL_CONFIGS])
    proxies = {"BOPs": np.array([bops(b) for b in ALL_CONFIGS], float),
               "weight_bits": np.array([wmem_bits(b) for b in ALL_CONFIGS], float)}
    out = {"n_configs": int(keep.sum()), "fps_target": args.fps, "rate": args.rate}
    for pname, pp in POWER.items():
        min_lut = np.full(n, np.inf)
        e_c = np.full(n, np.inf)
        e_r = np.full(n, np.inf)
        for f in F_SET:
            fps, ec, p_idle, e_dyn = energies(rows, pp, f)
            er = np.where(fps >= args.rate, p_idle / args.rate + e_dyn, np.inf)
            np.minimum.at(e_c, cfg, ec)
            np.minimum.at(e_r, cfg, er)
            if f == 100.0:
                lut_ok = np.where(fps >= args.fps, rows[:, 4], np.inf)
                np.minimum.at(min_lut, cfg, lut_ok)
        res = {}
        for cname, cost in [("min_LUT_at_target", min_lut), ("E_continuous", e_c),
                            ("E_fixed_rate", e_r)]:
            m = keep & np.isfinite(cost)
            res[cname] = {"n": int(m.sum())}
            for prox, pv in proxies.items():
                tau = kendalltau(pv[m], cost[m]).correlation
                rho = spearmanr(pv[m], cost[m]).correlation
                best_proxy = np.flatnonzero(m)[np.argmin(pv[m])]
                rank = int((cost[m] < cost[best_proxy]).sum())
                res[cname][prox] = {"kendall_tau": round(float(tau), 3),
                                    "spearman_rho": round(float(rho), 3),
                                    "proxy_best_config": list(ALL_CONFIGS[best_proxy]),
                                    "its_cost_rank": rank,
                                    "cost_ratio_vs_best": round(float(cost[best_proxy] /
                                                                      cost[m].min()), 3)}
        out[pname] = res
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
