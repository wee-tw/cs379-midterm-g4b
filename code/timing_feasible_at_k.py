"""Timing Feasible@K of a timing-blind cost model (implementation_notes.md 12.38).

The v2.1 cost tables are resource-calibrated but carry no timing: the search assumes every
design runs at up to 125 MHz. Here each design gets its clock class from the measured node
timing (timing_safe.node_class, RTL periods with the clock given to synthesis), and for each
accuracy margin we report, among the timing-blind model's K lowest-energy designs:
  feasible   the share whose class reaches the clock the model assumed for it
  optimism   how much lower the model's energy is than the same design at its class clock
and the energy of the model's pick at its real clock against the timing-aware optimum.

  python timing_feasible_at_k.py --tables cost/tables_calv2.json --points cost/points_calv2.npz \
      --rtl-timing rtl_all_constrained.json --acc cost/acc_model_3rep.json [--aware-opt merged.json]
"""

import argparse
import json

import numpy as np

import energy_search as es
from cost_tables import contexts
from timing_safe import node_class, rtl_lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--rtl-timing", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--aware-opt", help="timing_safe.py sweep output (timing-aware optimum per margin)")
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--unsafe-ns", type=float, default=13.14,
                    help="period for nodes timing_safe.py calls unsafe (wide folded MVAU_hls), only "
                         "to put a number on the energy of such a pick: the slowest one measured")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--margins", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 8.0])
    args = ap.parse_args()
    from calib_v2 import annotate
    table = annotate(json.load(open(args.tables))["table"])
    rtl = rtl_lookup(args.rtl_timing)
    misses = __import__("collections").Counter()
    rec_cls = {k: min(node_class(n, rtl, misses) for n in rec["nodes"]) for k, rec in table.items()}
    rows = np.load(args.points)["rows"]
    cfg = rows[:, 0].astype(int)

    def cls_of(i):
        b = es.ALL_CONFIGS[cfg[i]]
        mode, T = es.MODES[int(rows[i, 1])], int(rows[i, 2])
        return min(rec_cls[f"{mode}|{T}|{g}|{w}|{a}|{p}"] for g, (w, a, p) in enumerate(contexts(list(b))))

    p = es.POWER[args.power]
    es.set_fmax(125.0)
    per_f = np.stack([es.energies(rows, p, f)[1] for f in es.F_SET])
    e_blind = per_f.min(axis=0)
    f_blind = np.array(es.F_SET)[per_f.argmin(axis=0)]
    a = json.load(open(args.acc))
    acc = np.array([a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS])
    aware = json.load(open(args.aware_opt)) if args.aware_opt else None
    clocks = sorted(set(es.F_SET_FULL) | {125.0, 111.11, 100.0, 90.91, 83.33, 76.92, 71.43, 66.67, 50.0})
    print(f"power {args.power}; timing-blind clock set {es.F_SET}")
    for m in args.margins:
        thr = a["reference_acc"] - m
        ok = np.flatnonzero(acc[cfg] >= thr)
        order = ok[np.argsort(e_blind[ok])]
        line = [f"margin {m:g}:"]
        for k in args.k:
            top = order[:k]
            c = np.array([cls_of(i) for i in top])
            line.append(f"Feasible@{k} {100 * np.mean(c >= f_blind[top] - 1e-9):.0f}%")
        i0 = order[0]
        c0 = cls_of(i0)
        if c0 == 0:   # an unsafe node: take the slowest period measured for one
            from timing_safe import clock_class
            c0 = clock_class(args.unsafe_ns)
        e_real = min(es.energies(rows[i0:i0 + 1], p, f)[1][0] for f in clocks if f <= c0 + 1e-9)
        b0 = "".join(map(str, es.ALL_CONFIGS[cfg[i0]]))
        line.append(f"| pick {b0} {es.MODES[int(rows[i0, 1])]} T{int(rows[i0, 2])}: model {e_blind[i0] * 1e3:.4f} mJ "
                    f"at {f_blind[i0]:g} MHz, class {c0:g} MHz -> {e_real * 1e3:.4f} mJ "
                    f"(model optimistic by {100 * (1 - e_blind[i0] / e_real):.1f}%)")
        if aware:
            ea = aware[str(m)][args.power]["continuous"]["A_prime"]["E_mJ"]
            line.append(f"| vs timing-aware optimum {ea:.4f} mJ: {100 * (e_real * 1e3 / ea - 1):+.1f}%")
        print(" ".join(line))


if __name__ == "__main__":
    main()
