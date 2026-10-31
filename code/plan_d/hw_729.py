"""Handover 5 for Plan D: results/plan_d/hw_729.csv, the hardware columns of the 729 configurations
(G1-G3 at 2/4/8 bits, FC fixed at W8A8, i.e. our 8-digit configurations ending in 88).

One row per configuration: the design with the lowest nominal energy per frame among the final
design points (code/cost/points_tsr_v1218_f*.npz: v2.1 calibration, timing-safe refold RTL SIMD <= 12
/ HLS SIMD <= 18, every design at the PYNQ clock class its slowest node meets; all folding targets
and implementation modes). The first 18 columns follow Plan D 8.5; the rest are documented in
results/plan_d/hw_729_README.md.

  * main values: 9 clock classes, budget LUT 80 %, BRAM18 70 %, DSP 220
  * f100_*: fixed 100 MHz with Plan D's budget (LUT 70 %)
  * margin10_*: clock classes with 10 % routing margin (points_tsr_v1218_m0.1_f*.npz)

Energy E = (P0/f + k_f + (k_fU + k_act) U) x cycles, U = LUT/53200 + 0.25 BRAM/280 + 0.25 DSP/220
(energy_search.py). Split: static = P0/f x cycles; clock = k_f x cycles; logic = (k_fU + k_act) x
(LUT and DSP terms of U) x cycles; memory = the same for the BRAM term. Latency = FINN critical path
cycles (latency_check.py) / f.

  python plan_d/hw_729.py        (from code/; about 1 minute)
"""

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_search as es  # noqa: E402
from assemble_costs import assemble  # noqa: E402
from latency_check import crit_from_nodes  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
COST = os.path.join(ROOT, "code", "cost")
CLASSES = (125.0, 111.11, 100.0, 90.91, 83.33, 76.92, 71.43, 66.67, 50.0)
BUDGET_OURS = {"LUT": 53200 * 0.8, "BRAM": 280 * 0.7, "DSP": 220}
BUDGET_PLAN_D = {"LUT": 53200 * 0.7, "BRAM": 280 * 0.7, "DSP": 220}
SETS = {  # name: (points file pattern, tables file pattern, classes, budget)
    "main": ("points_tsr_v1218_f%g.npz", "tables_calv2_tsr_v1218_f%g.json.gz", CLASSES, BUDGET_OURS),
    "f100": ("points_tsr_v1218_f%g.npz", "tables_calv2_tsr_v1218_f%g.json.gz", (100.0,), BUDGET_PLAN_D),
    "margin10": ("points_tsr_v1218_m0.1_f%g.npz", "tables_tsr_v1218_m0.1_f%g.json.gz", CLASSES, BUDGET_OURS),
}


def bops_group_a(b):
    """BOPs when every layer uses its own group's A (the stem too) instead of its input's."""
    w, a = b[0::2], b[1::2]
    return sum(macs * w[g] * a[g] for _, _, macs, g, _ in es.LAYERS)


def select(name, configs, idx):
    """Per configuration: (f, row) of the lowest nominal energy design within the budget."""
    pat, _, classes, budget = SETS[name]
    best = {}
    for f in classes:
        rows = np.load(os.path.join(COST, pat % f))["rows"]
        ok = (rows[:, 4] <= budget["LUT"]) & (rows[:, 5] <= budget["BRAM"]) & (rows[:, 6] <= budget["DSP"])
        e = es.energies(rows, es.POWER["nominal"], f)[1]
        cfg = rows[:, 0].astype(int)
        for c in configs:
            m = ok & (cfg == idx[c])
            if m.any():
                j = np.flatnonzero(m)[np.argmin(e[m])]
                if c not in best or e[j] < best[c][2]:
                    best[c] = (f, rows[j].copy(), float(e[j]))
    return best


def describe(name, best):
    """Hardware columns of the selected designs (latency from the class's cost table)."""
    _, tpat, _, _ = SETS[name]
    out, tables = {}, {}
    for c, (f, r, _) in best.items():
        if f not in tables:
            with gzip.open(os.path.join(COST, tpat % f), "rt") as fh:
                tables[f] = json.load(fh)["table"]
        b = list(es.ALL_CONFIGS[int(r[0])])
        nodes = assemble(tables[f], b, es.MODES[int(r[1])], int(r[2]))
        crit = crit_from_nodes(nodes)[0]
        mvau = sum(n["cycles"] for n in nodes if n["op"].startswith("MVAU"))
        cyc, lut, bram, dsp = r[3], r[4], r[5], r[6]
        assert cyc == max(n["cycles"] for n in nodes), (c, name)
        d = dict(f=f, mode=es.MODES[int(r[1])], T=int(r[2]), ii=int(cyc), crit=int(crit), LUT=int(lut),
                 BRAM=int(bram), DSP=int(dsp), lat_ms=crit / (f * 1e3), lat_planD_ms=mvau / (f * 1e3),
                 fps=f * 1e6 / cyc)
        for pn in ("nominal", "z2_anchor"):
            d["E_" + pn] = float(es.energies(r[None, :], es.POWER[pn], f)[1][0]) * 1e3
        p = es.POWER["nominal"]
        u_logic = es.UTIL_W["LUT"] * lut / es.DEVICE["LUT"] + es.UTIL_W["DSP"] * dsp / es.DEVICE["DSP"]
        u_bram = es.UTIL_W["BRAM_18K"] * bram / es.DEVICE["BRAM_18K"]
        d["E_static"] = p["P0"] / f * cyc * 1e-3
        d["E_clock"] = p["k_f"] * cyc * 1e-3
        d["E_logic"] = (p["k_fU"] + p["k_act"]) * u_logic * cyc * 1e-3
        d["E_mem"] = (p["k_fU"] + p["k_act"]) * u_bram * cyc * 1e-3
        assert abs(d["E_static"] + d["E_clock"] + d["E_logic"] + d["E_mem"] - d["E_nominal"]) < 1e-9, c
        out[c] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "plan_d"))
    args = ap.parse_args()
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    c729 = sorted(c for c in idx if c.endswith("88"))
    res = {name: describe(name, select(name, c729, idx)) for name in SETS}

    r6 = lambda x: round(x, 6)  # noqa: E731
    head = ["g1_w", "g1_a", "g2_w", "g2_a", "g3_w", "g3_a", "LUT", "DSP", "BRAM18", "latency_ms", "fps",
            "energy_mJ_k14", "energy_mJ_k7", "E_static_mJ", "E_compute_mJ", "E_mem_mJ", "model_size_bits", "BOPs",
            "config8", "feasible", "impl_mode", "fold_target_cycles", "f_MHz", "ii_cycles", "latency_cycles",
            "energy_mJ_nominal", "energy_mJ_z2_anchor", "E_clock_mJ", "E_logic_mJ", "latency_ms_planD_def",
            "BOPs_group_act"]
    for v in ("f100", "margin10"):
        head += [v + "_" + k for k in ("feasible", "f_MHz", "LUT", "DSP", "BRAM18", "latency_ms", "fps",
                                       "energy_mJ_nominal", "energy_mJ_z2_anchor")]
    path = os.path.join(args.out, "hw_729.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(head)
        for c in c729:
            b = [int(x) for x in c]
            row = b[:6]
            m = res["main"].get(c)
            if m:
                row += [m["LUT"], m["DSP"], m["BRAM"], r6(m["lat_ms"]), round(m["fps"], 2), r6(m["E_nominal"]),
                        r6(m["E_z2_anchor"]), r6(m["E_static"]), r6(m["E_clock"] + m["E_logic"]), r6(m["E_mem"])]
            else:
                row += [""] * 10
            row += [es.wmem_bits(b), es.bops(b), c, int(bool(m))]
            row += ([m["mode"], m["T"], m["f"], m["ii"], m["crit"], r6(m["E_nominal"]), r6(m["E_z2_anchor"]),
                     r6(m["E_clock"]), r6(m["E_logic"]), r6(m["lat_planD_ms"])] if m else [""] * 10)
            row.append(bops_group_a(b))
            for v in ("f100", "margin10"):
                d = res[v].get(c)
                row += ([1, d["f"], d["LUT"], d["DSP"], d["BRAM"], r6(d["lat_ms"]), round(d["fps"], 2),
                         r6(d["E_nominal"]), r6(d["E_z2_anchor"])] if d else [0] + [""] * 8)
            wr.writerow(row)

    # self-checks asked for in the handover (2.3) and cross-checks against other files
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    lines = ["hw_729.csv checks (code/plan_d/hw_729.py)", ""]
    bits = {(r["g1_w"], r["g1_a"], r["g2_w"], r["g2_a"], r["g3_w"], r["g3_a"]) for r in rows}
    lines.append("rows %d; distinct bit settings %d; bit values %s; feasible: main %d, f100 %d, margin10 %d"
                 % (len(rows), len(bits), sorted({v for t in bits for v in t}),
                   sum(r["feasible"] == "1" for r in rows), sum(r["f100_feasible"] == "1" for r in rows),
                   sum(r["margin10_feasible"] == "1" for r in rows)))
    by_lut = sorted(rows, key=lambda r: -int(r["LUT"] or 0))
    lines.append("largest LUT (main): " + ", ".join("%s %s (%s, %s DSP)" % (r["config8"], r["LUT"], r["impl_mode"], r["DSP"])
                                                    for r in by_lut[:3]))
    w8 = next(r for r in rows if r["config8"] == "88888888")
    lines.append("88888888 (main): %s LUT, %s DSP, %s BRAM18, %s, %s MHz, %s mJ"
                 % (w8["LUT"], w8["DSP"], w8["BRAM18"], w8["impl_mode"], w8["f_MHz"], w8["energy_mJ_k14"]))
    lo = min(rows, key=lambda r: int(r["BOPs"]))
    lines.append("smallest BOPs: %s %s" % (lo["config8"], lo["BOPs"]))
    clk = {}
    for r in rows:
        clk[r["f_MHz"]] = clk.get(r["f_MHz"], 0) + 1
    lines.append("main clock classes used: " + ", ".join("%s MHz x %d" % kv for kv in sorted(clk.items())))
    # f100 column vs results/plan_d/rq2_729_energies.csv (plan_d_checks.py, same points and budget)
    ref = {r["config8"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "plan_d", "rq2_729_energies.csv")))}
    worst = max(abs(float(r["f100_energy_mJ_nominal"]) - float(ref[r["config8"]]["E_nominal_mJ_f100_planD"])) for r in rows)
    lines.append("f100 energy vs rq2_729_energies.csv: max |difference| %.2g mJ" % worst)
    # latency at 100 MHz vs the latency file written by latency_check.py for the same points
    lat = np.load(os.path.join(COST, "latency_tsr_v1218_f100.npz"))["latency_cycles"]
    pts = np.load(os.path.join(COST, "points_tsr_v1218_f100.npz"))["rows"]
    key = {(int(p[0]), int(p[1]), int(p[2]), int(p[4]), int(p[5]), int(p[6])): int(l) for p, l in zip(pts, lat)}
    bad = 0
    for c, d in res["f100"].items():
        k = (idx[c], es.MODES.index(d["mode"]), d["T"], d["LUT"], d["BRAM"], d["DSP"])
        bad += key.get(k) != d["crit"]
    lines.append("f100 latency cycles vs latency_tsr_v1218_f100.npz: %d of %d differ" % (bad, len(res["f100"])))
    for c in ("44444488", "22222288", "84242488"):
        r = next(x for x in rows if x["config8"] == c)
        lines.append("spot check %s: %s LUT, %s DSP, %s BRAM18, %s MHz, %s mJ nominal, %s mJ z2_anchor, "
                     "split %s + %s + %s + %s" % (c, r["LUT"], r["DSP"], r["BRAM18"], r["f_MHz"], r["energy_mJ_k14"],
                                                  r["energy_mJ_k7"], r["E_static_mJ"], r["E_clock_mJ"], r["E_logic_mJ"], r["E_mem_mJ"]))
    out = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "hw_729_checks.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
