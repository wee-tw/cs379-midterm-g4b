"""Timing classes of the design space (implementation_notes.md 12.35-12.36, 12.41).

Every node gets the highest PL clock of F_CLASSES (FCLK0 settings 1000/n MHz a PYNQ-Z2 can
generate) whose period is at least its post-synthesis critical path (Vivado / Vitis HLS 2022.2,
xc7z020clg400-1). Out-of-context place and route of 24 critical nodes (route_timing.py) put 23
in the same class; the exception, a folded MVAU_hls at SIMD 24, routed at 9.45 ns (100 MHz),
which --hls-routed-simd 18 applies to every folded MVAU_hls wider than SIMD 18:
  MVAU_hls (330 runs, hls_timing_scan.py), with SF = MW / SIMD the number of input folds:
    SF == 1                every run met 8 ns (worst 7.63 ns)                  -> 125 MHz
    SF > 1, SIMD <= 27     worst 8.80 ns (every run met 10 ns, 8 of 9 missed 8 ns) -> 111.11 MHz
    SF > 1, SIMD >= 28     unsafe (21 of 36 missed 10 ns); cost_tables.py --ts-refold removes them
  MVAU_rtl: the measured period of the same configuration (rtl_timing.py on the 53 audited
    units); a configuration without a measurement uses rtl_model (same core, nearest
    measured SIMD at or above its own)
  other nodes: 125 MHz (66 other HLS nodes <= 7.08 ns; largest Thresholding_rtl,
    ConvolutionInputGenerator_rtl and FMPadding_rtl <= 7.10 ns)
A record's class is its slowest node's, a design's is its slowest record's.

  tables  writes <prefix>_f<class>.json with the records of at least that class
  sweep   energy_search's selections on each class's points with the clocks capped at the
          class, merged per selection: lower energy for A' and B1, higher FPS for B3 and B4,
          B2 (FINN default folding at 100 MHz) from the 100 MHz class
  rank    per margin, the lowest-energy configurations with their measured accuracy
  table   the main table with measured accuracy: verified A' (fine-tunes pass on average),
          the sweep's B1 and B2 with their measured status, B1 re-picked among the uniform
          configurations whose measurement passes, and B2 re-picked along its BOPs order
  tables --margin / --op-margin lengthen the post-synthesis periods (routing sensitivity)

  python timing_safe.py tables --tables cost/tables_calv2_tsr.json --rtl-timing rtl_timing.json \
      --prefix cost/tables_calv2_tsr
  python timing_safe.py sweep --points 125=points_f125.npz 100=points_f100.npz 66.67=... 50=... \
      --acc cost/acc_model_3rep.json --out ../results/search/threshold_sweep_timing.json
"""

import argparse
import collections
import copy
import json

import numpy as np

import energy_search as es

F_CLASSES = (125.0, 111.11, 100.0, 90.91, 83.33, 76.92, 71.43, 66.67, 50.0)   # 1000/n MHz
SIMD_SAFE_100 = 27   # widest folded MVAU_hls seen to meet 10 ns (worst 8.80 ns)
MARGIN = 0.0         # tables --margin: every post-synthesis period * (1 + MARGIN), for routing
OP_MARGIN = {}       # tables --op-margin: per node kind (MVAU_rtl, MVAU_hls, other), overrides MARGIN
HLS_ROUTED = None    # tables --hls-routed-simd: folded MVAU_hls wider than this (up to 27) take the
                     # post-route period of the SIMD-24 unit (route_timing.py, 9.45 ns) instead


def clock_class(period_ns, kind=None):
    m = OP_MARGIN.get(kind, MARGIN)
    return next((f for f in F_CLASSES if 1000.0 / f >= period_ns * (1 + m)), 0.0)


def bits(dtype):
    return int("".join(ch for ch in dtype if ch.isdigit()))


def rtl_lookup(path):
    """(MW, MH, PE, SIMD, W, A) -> measured post-synthesis period of an MVAU_rtl unit."""
    out = {}
    for r in json.load(open(path)):
        if r["op"] != "MVAU_rtl" or r["achieved_ns"] is None:
            continue
        a = r["attrs"]
        k = (int(a["MW"]), int(a["MH"]), int(a["PE"]), int(a["SIMD"]),
             bits(a["weightDataType"]), bits(a["inputDataType"]))
        out[k] = max(out.get(k, 0.0), r["achieved_ns"])
    return out


def node_class(n, rtl, misses):
    if n["op"] == "MVAU_hls":
        mw, simd = n["shape"]["MW"], n["fold"]["SIMD"]
        if simd == mw:
            return clock_class(7.63, "MVAU_hls")
        if simd > SIMD_SAFE_100:
            return 0.0
        return clock_class(9.45 if HLS_ROUTED and simd > HLS_ROUTED else 8.80, "MVAU_hls")
    if n["op"] == "MVAU_rtl":
        s, f = n["shape"], n["fold"]
        k = (s["MW"], s["MH"], f["PE"], f["SIMD"], s["W"], s["A"])
        if k in rtl:
            return clock_class(rtl[k], "MVAU_rtl")
        misses[k] += 1
        return clock_class(rtl_model(rtl, s["W"] <= 4 and s["A"] <= 4, f["SIMD"]), "MVAU_rtl")
    return clock_class(7.10, "other")


def rtl_model(rtl, core4, simd):
    """Period of an unmeasured MVAU_rtl: the slowest measured unit of the same core at the
    smallest measured SIMD >= simd (the LUT lane adder grows with SIMD), or at the widest
    measured SIMD if simd is wider than all of them."""
    same = [(s, p) for (mw, mh, pe, s, w, a), p in rtl.items() if (w <= 4 and a <= 4) == core4]
    if not same:
        return float("inf")
    up = [s for s, _ in same if s >= simd]
    at = min(up) if up else max(s for s, _ in same)
    return max(p for s, p in same if s == at)


def cmd_tables(args):
    from calib_v2 import annotate
    global MARGIN, HLS_ROUTED
    MARGIN = args.margin
    HLS_ROUTED = args.hls_routed_simd
    OP_MARGIN.update((k, float(v)) for k, v in (s.split("=", 1) for s in args.op_margin))
    full = json.load(open(args.tables))
    rtl = rtl_lookup(args.rtl_timing)
    misses = collections.Counter()
    shaped = annotate(copy.deepcopy(full["table"]))
    cls = {k: min(node_class(n, rtl, misses) for n in rec["nodes"]) for k, rec in shaped.items()}
    count = collections.Counter((k.split("|")[0], cls[k]) for k in cls)
    for mode in sorted({m for m, _ in count}):
        print(f"{mode:9s} records by class: " + ", ".join(
            f"{c:g} MHz {count[(mode, c)]}" for c in F_CLASSES + (0.0,) if count[(mode, c)]))
    print(f"MVAU_rtl configurations without a measurement: {len(misses)} "
          f"({sum(misses.values())} node entries)")
    for c in F_CLASSES:
        out = {**full, "table": {k: v for k, v in full["table"].items() if cls[k] >= c},
               "timing_class": {"fmax_MHz": c, "simd_safe_100": SIMD_SAFE_100, "margin": MARGIN,
                                "op_margin": OP_MARGIN, "hls_routed_simd": HLS_ROUTED}}
        path = f"{args.prefix}_f{c:g}.json"
        json.dump(out, open(path, "w"))
        print("wrote", path, len(out["table"]), "of", len(full["table"]), "records")


def merge(a, b):
    """Best of two selections of the same baseline: lower energy for the energy optima,
    higher FPS for the fastest-design baselines (B3, B4)."""
    if a is None or b is None:
        return a or b
    if a["name"] in ("B3_sequential", "B4_latency"):
        return a if (a["FPS"], -a["U"]) >= (b["FPS"], -b["U"]) else b
    ea, eb = a.get("E"), b.get("E")
    if ea is None or eb is None:
        return a if eb is None else b
    return a if ea <= eb else b


def run(rows, acc, thr, p, fmax):
    if len(rows) == 0:
        return {"continuous": {}, "fixed_rate": collections.defaultdict(dict)}
    es.set_fmax(fmax)
    return es.search(rows, acc, thr, p)


def cmd_sweep(args):
    sets = []
    for item in args.points:
        cap, path = item.split("=", 1)
        sets.append((float(cap), np.load(path)["rows"]))
    a = json.load(open(args.acc))
    acc = [a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS]
    out = {}
    for m in args.margins:
        thr = a["reference_acc"] - m
        res = {"threshold": round(thr, 2)}
        for pname in args.powers:
            p = es.POWER[pname]
            runs = [(cap, run(rows, acc, thr, p, cap)) for cap, rows in sets]
            r = {}
            for regime in ["continuous"] + [f"fixed_{R}" for R in args.rates]:
                get = (lambda s: s["continuous"]) if regime == "continuous" else (
                    lambda s, R=regime[6:]: s["fixed_rate"][R])
                ekey = "E_continuous_mJ" if regime == "continuous" else "E_fixed_mJ"
                sel = {}
                for name in set().union(*(get(s) for _, s in runs)):
                    best = None
                    for cap, s in sorted(runs, key=lambda r: -r[0]):
                        d = get(s).get(name)
                        if d is None:
                            continue
                        if name == "B2_proxy":   # FINN's 100 MHz, else the fastest class below it
                            if cap <= 100.0:
                                best = {**d, "name": name, "E": d.get(ekey), "class_MHz": cap}
                                break
                            continue
                        best = merge(best, {**d, "name": name, "E": d.get(ekey), "class_MHz": cap})
                    if best is not None:
                        sel[name] = best
                if "A_prime" not in sel:   # nothing meets the threshold (or the rate)
                    r[regime] = None
                    continue
                ea = sel["A_prime"]["E"]
                r[regime] = {k: {"rel_to_A": (None if d["E"] is None or ea is None
                                              else round(d["E"] / ea - 1, 4)),
                                 "E_mJ": d["E"], "bits": "".join(map(str, d["bits"])),
                                 "mode": d["mode"], "f_MHz": d["f_MHz"], "FPS": d["FPS"],
                                 "LUT": d["LUT"], "BRAM18": d["BRAM18"], "DSP": d["DSP"],
                                 "T_target": d["T_target"], "class_MHz": d["class_MHz"]}
                             for k, d in sorted(sel.items())}
            res[pname] = r
        out[str(m)] = res
        print(f"== margin {m} (threshold {thr:.2f})")
        for pname in args.powers:
            c = res[pname]["continuous"]
            if c is None:
                print(f"  {pname:9s} no design meets the threshold")
                continue
            ap = c["A_prime"]
            print(f"  {pname:9s} A'={ap['bits']} {ap['mode']} T{ap['T_target']} {ap['f_MHz']:g} MHz "
                  f"{ap['E_mJ']} mJ | " + ", ".join(
                      f"{k.split('_')[0]} {100 * d['rel_to_A']:+.1f}% ({d['bits']} {d['mode']} {d['f_MHz']:g})"
                      for k, d in c.items() if k != "A_prime" and d["rel_to_A"] is not None))
    json.dump(out, open(args.out, "w"), indent=1)


def measured(dirs):
    """bits string -> final accuracies of the 30-epoch candidate fine-tunes (ft30_cand_*)."""
    import glob
    import os
    out = collections.defaultdict(list)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "ft30_cand_*.json")):
            j = json.load(open(f))
            out["".join(map(str, j["args"]["bits"]))].append(j["final_acc"])
    return out


def best_per_config(points, p):
    """Lowest continuous-inference energy of every configuration over the timing classes, and
    the (mode, T, clock) that reaches it."""
    best = np.full(len(es.ALL_CONFIGS), np.inf)
    info = {}
    for item in points:
        cap, path = item.split("=", 1)
        rows = np.load(path)["rows"]
        if len(rows) == 0:
            continue
        es.set_fmax(float(cap))
        for f in es.F_SET:
            e = es.energies(rows, p, f)[1]
            cfg = rows[:, 0].astype(int)
            for i in np.flatnonzero(e < best[cfg]):
                if e[i] < best[cfg[i]]:
                    best[cfg[i]] = e[i]
                    info[cfg[i]] = (es.MODES[int(rows[i, 1])], int(rows[i, 2]), f)
    return best, info


def cmd_rank(args):
    """Per margin, the lowest-energy configurations (each at the best clock its class allows)
    with the model's predicted accuracy and the measured QAT accuracy where it exists."""
    a = json.load(open(args.acc))
    acc = np.array([a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS])
    meas = measured(args.logs)
    best, info = best_per_config(args.points, es.POWER[args.power])
    for m in args.margins:
        thr = a["reference_acc"] - m
        ok = np.flatnonzero((acc >= thr) & np.isfinite(best))
        order = ok[np.argsort(best[ok])][:args.top]
        print(f"== margin {m:g} (threshold {thr:.2f}), {args.power}")
        for c in order:
            b = "".join(map(str, es.ALL_CONFIGS[c]))
            v = meas.get(b, [])
            ms = (f"measured {np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f} (n={len(v)})"
                  + (" PASS" if np.mean(v) >= thr else " FAIL")) if v else "not trained"
            mode, T, f = info[c]
            print(f"   {b} {mode:8s} T{T:<7d} {f:6.2f} MHz  E {best[c] * 1e3:.4f} mJ  "
                  f"(+{100 * (best[c] / best[order[0]] - 1):.1f}%)  pred {acc[c]:.2f}  {ms}")


def cmd_table(args):
    """Main table with measured accuracy. Per margin, the verified A' is the lowest-energy
    configuration whose fine-tunes (at least --min-n seeds) pass the threshold on average.
    B1 and B2 are the sweep's picks (selected with the accuracy model) with their measured
    status. B1_measured re-picks the best uniform configuration among the measured ones that
    pass; B2_measured walks B2's BOPs order and takes the first configuration whose fine-tunes
    pass (at least --min-n seeds), stopping at one that is not trained enough yet; B3_measured
    is the same bits with our folding (the sweep's B3: fastest design, energy at its clock)."""
    a = json.load(open(args.acc))
    acc = np.array([a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS])
    meas = measured(args.logs)
    for item in args.extra:   # e.g. the W4A4 base model, which has no ft30 log
        b, v = item.split("=", 1)
        meas[b] += [float(x) for x in v.split(",")]
    sweep = json.load(open(args.sweep))
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    low = sorted(((float(cap), path) for cap, path in (s.split("=", 1) for s in args.points)
                  if float(cap) <= 100.0), reverse=True)
    sets = [(float(cap), np.load(path)["rows"]) for cap, path in (s.split("=", 1) for s in args.points)]
    out = {}

    def b3_energy(b, p):
        """B3's design for bits b as in the sweep: our folding (any mode but lut_finn), the
        fastest design at the cap of its timing class, ties to the fewest resources."""
        key, e = None, None
        for cap, rows in sets:
            if len(rows) == 0:
                continue
            mask = (rows[:, 0].astype(int) == idx[b]) & (rows[:, 1] != es.MODES.index("lut_finn"))
            if not mask.any():
                continue
            fps, e_c, _, _ = es.energies(rows, p, cap)
            U = es.utilisation(rows[:, 4], rows[:, 5], rows[:, 6])
            for i in np.flatnonzero(mask):
                if key is None or (fps[i], -U[i]) > key:
                    key, e = (fps[i], -U[i]), float(e_c[i])
        return e

    def b2_energy(b, p):
        """B2's design for bits b as in the sweep: FINN default folding (lut_finn), fastest,
        at 100 MHz or at the fastest timing class below it that has the design."""
        for cap, path in low:
            rows = np.load(path)["rows"]
            if len(rows) == 0:
                continue
            mask = (rows[:, 0].astype(int) == idx[b]) & (rows[:, 1] == es.MODES.index("lut_finn"))
            if mask.any():
                fps, e_c, _, _ = es.energies(rows, p, cap)
                return float(e_c[int(np.argmax(np.where(mask, fps, -np.inf)))])
        return None

    for pname in args.powers:
        best, info = best_per_config(args.points, es.POWER[pname])
        print(f"== {pname}")
        for m in args.margins:
            thr = a["reference_acc"] - m

            def passes(b, n=1):
                v = meas.get(b, [])
                return len(v) >= n and np.mean(v) >= thr

            def status(b):
                v = meas.get(b, [])
                return (f"{np.mean(v):.2f} n={len(v)} " + ("PASS" if passes(b) else "FAIL")) if v else "not trained"

            ok = [b for b in meas if b in idx and passes(b, args.min_n) and np.isfinite(best[idx[b]])]
            if not ok:
                print(f"  margin {m:g}: no measured configuration passes")
                continue
            av = min(ok, key=lambda b: best[idx[b]])
            ea = best[idx[av]]
            c = sweep[str(m)][pname]["continuous"]
            uni = [b for b in meas if b in idx and es.is_uniform(es.ALL_CONFIGS[idx[b]])
                   and passes(b) and np.isfinite(best[idx[b]])]
            b1m = min(uni, key=lambda b: best[idx[b]]) if uni else None
            mode, T, f = info[idx[av]]
            row = {"threshold": round(thr, 2), "A_verified": {
                       "bits": av, "mode": mode, "T_target": T, "f_MHz": f, "E_mJ": round(ea * 1e3, 4),
                       "measured": status(av), "rel_to_model_opt": round(ea * 1e3 / c["A_prime"]["E_mJ"] - 1, 4)}}
            for k in ("B1_uniform", "B2_proxy"):
                d = c[k]
                row[k] = {**d, "rel_to_A": round(d["E_mJ"] / (ea * 1e3) - 1, 4), "measured": status(d["bits"])}
            if b1m is not None:
                row["B1_measured"] = {"bits": b1m, "E_mJ": round(best[idx[b1m]] * 1e3, 4),
                                      "rel_to_A": round(best[idx[b1m]] / ea - 1, 4), "measured": status(b1m)}
            skipped = []
            for c2 in sorted(np.flatnonzero(acc >= thr), key=lambda c: (es.bops(es.ALL_CONFIGS[c]), -acc[c])):
                b = "".join(map(str, es.ALL_CONFIGS[c2]))
                if len(meas.get(b, [])) < args.min_n:
                    row["B2_measured"] = {"bits": None, "next_untrained": b, "failed": skipped}
                    break
                if passes(b, args.min_n):
                    e2 = b2_energy(b, es.POWER[pname])
                    row["B2_measured"] = {"bits": b, "E_mJ": None if e2 is None else round(e2 * 1e3, 4),
                                          "rel_to_A": None if e2 is None else round(e2 / ea - 1, 4),
                                          "measured": status(b), "failed": skipped}
                    e3 = b3_energy(b, es.POWER[pname])   # the same bits with our folding (H3)
                    row["B3_measured"] = {"bits": b, "E_mJ": None if e3 is None else round(e3 * 1e3, 4),
                                          "rel_to_A": None if e3 is None else round(e3 / ea - 1, 4),
                                          "measured": status(b)}
                    break
                skipped.append(b)
            out.setdefault(str(m), {})[pname] = row
            print(f"  margin {m:g}: A'v {av} {mode} T{T} {f:g} MHz {ea * 1e3:.4f} mJ "
                  f"(model opt {c['A_prime']['bits']} {100 * row['A_verified']['rel_to_model_opt']:+.1f}%) "
                  f"[{status(av)}]")
            for k in ("B1_uniform", "B2_proxy", "B1_measured"):
                if k in row:
                    d = row[k]
                    print(f"      {k:11s} {d['bits']} {100 * d['rel_to_A']:+.1f}%  [{d['measured']}]")
            d = row.get("B2_measured")
            if d and d["bits"]:
                print(f"      B2_measured {d['bits']} {100 * d['rel_to_A']:+.1f}%  [{d['measured']}]"
                      f"  (failed before it: {d['failed'] or '-'})")
                d3 = row["B3_measured"]
                if d3["rel_to_A"] is not None:
                    print(f"      B3_measured {d3['bits']} {100 * d3['rel_to_A']:+.1f}%  (same bits, our folding)")
            elif d:
                print(f"      B2_measured needs {d['next_untrained']} trained (failed before it: {d['failed'] or '-'})")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["tables", "sweep", "rank", "table"])
    p.add_argument("--logs", nargs="+", default=[], help="rank, table: folders with ft30_cand_*.json logs")
    p.add_argument("--extra", nargs="+", default=[],
                   help="table: BITS=acc[,acc...] measured outside the ft30 logs (e.g. the W4A4 base)")
    p.add_argument("--min-n", type=int, default=3, help="table: seeds needed for the verified A'")
    p.add_argument("--sweep", help="table: merged sweep JSON (sweep --out)")
    p.add_argument("--power", default="nominal")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--tables")
    p.add_argument("--rtl-timing", help="tables: rtl_timing.py output(s) merged into one JSON list")
    p.add_argument("--margin", type=float, default=0.0,
                   help="tables: lengthen every post-synthesis period by this fraction (routing)")
    p.add_argument("--hls-routed-simd", type=int, default=None,
                   help="tables: folded MVAU_hls with SIMD above this get the routed 9.45 ns (100 MHz)")
    p.add_argument("--op-margin", nargs="+", default=[],
                   help="tables: per-kind routing margin, e.g. MVAU_rtl=0.2 MVAU_hls=0.02 other=0.05")
    p.add_argument("--prefix")
    p.add_argument("--points", nargs="+", help="sweep: <class MHz>=<points.npz> per class")
    p.add_argument("--acc")
    p.add_argument("--margins", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 8.0])
    p.add_argument("--powers", nargs="+", default=["nominal", "z2_anchor"])
    p.add_argument("--rates", nargs="+", default=["300", "3000"])
    p.add_argument("--out")
    args = p.parse_args()
    {"tables": cmd_tables, "sweep": cmd_sweep, "rank": cmd_rank, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
