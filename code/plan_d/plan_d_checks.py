"""Numbers in docs/handoff5_8_reply.md that come from local-only inputs, written out so the team
can trace them on GitHub (results/plan_d/).

Inputs:
  * the final cost points (timing-safe refold RTL <= 12 / HLS <= 18, clock classes from
    post-synthesis timing): code/cost/points_tsr_v1218_f{125,111.11,100}.npz (the default --points-dir).
  * results/accuracy/acc_model_3rep.json and results/audit_dataset/rtl_nodes.csv (on GitHub).
Outputs:
  * results/plan_d/rq2_729_energies.csv: one row per Plan D configuration (FC at W8A8) with model
    size, BOPs in two conventions and the lowest nominal energy per frame with Plan D's budget
    (LUT 70 %, BRAM18 70 %, DSP 220), at a fixed 100 MHz and at the best PYNQ clock class.
    The ranking numbers below can be recomputed from this file alone.
  * results/plan_d/plan_d_checks.txt: the summary quoted in the reply.

  python plan_d/plan_d_checks.py
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_search as es  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CLASSES = (125.0, 111.11, 100.0)
BUDGET_PLAN_D = {"LUT": 53200 * 0.7, "BRAM": 280 * 0.7, "DSP": 220}
BUDGET_OURS = {"LUT": 53200 * 0.8, "BRAM": 280 * 0.7, "DSP": 220}


def bops_group_a(b):
    """BOPs when every layer uses its own group's A (the stem too) instead of its input's."""
    w, a = b[0::2], b[1::2]
    return sum(macs * w[g] * a[g] for _, _, macs, g, _ in es.LAYERS)


def best_per_config(points_dir, classes, budget, configs, idx):
    """Lowest nominal energy (mJ) of each configuration over the given clock classes."""
    best = {}
    for f in classes:
        rows = np.load(os.path.join(points_dir, "points_tsr_v1218_f%g.npz" % f))["rows"]
        ok = (rows[:, 4] <= budget["LUT"]) & (rows[:, 5] <= budget["BRAM"]) & (rows[:, 6] <= budget["DSP"])
        e = es.energies(rows, es.POWER["nominal"], f)[1] * 1e3
        cfg = rows[:, 0].astype(int)
        for c in configs:
            m = ok & (cfg == idx[c])
            if m.any():
                j = np.where(m)[0][np.argmin(e[m])]
                if c not in best or e[j] < best[c][0]:
                    best[c] = (float(e[j]), f, es.MODES[int(rows[j, 1])], int(rows[j, 4]),
                               int(rows[j, 5]), int(rows[j, 6]))
    return best


def finn_bram(n, p, w):
    """FINN v0.10.1 MVAU bram_estimation (block RAM case), p = PE x SIMD."""
    omega, mw = n / p, p * w
    if mw == 1:
        return math.ceil(omega / 16384)
    if mw == 2:
        return math.ceil(omega / 8192)
    if mw <= 4:
        return math.ceil(omega / 4096) * math.ceil(mw / 4)
    if mw <= 9:
        return math.ceil(omega / 2048) * math.ceil(mw / 9)
    if mw <= 18 or omega > 512:
        return math.ceil(omega / 1024) * math.ceil(mw / 18)
    return math.ceil(omega / 512) * math.ceil(mw / 36)


def plan_d_bram(n, p, w):
    """Plan D 5.4 (4): the cheapest of the six BRAM18 shapes."""
    shapes = [(16384, 1), (8192, 2), (4096, 4), (2048, 9), (1024, 18), (512, 36)]
    return min(math.ceil(p * w / sw) * math.ceil(n / p / sd) for sd, sw in shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-dir", default=os.path.join(ROOT, "code", "cost"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "plan_d"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    c729 = sorted(c for c in idx if c.endswith("88"))
    lines = ["Plan D checks (code/plan_d/plan_d_checks.py); energy = nominal parameters, per frame", ""]

    e100 = best_per_config(args.points_dir, (100.0,), BUDGET_PLAN_D, c729, idx)
    ecls = best_per_config(args.points_dir, CLASSES, BUDGET_PLAN_D, c729, idx)
    ours = best_per_config(args.points_dir, CLASSES, BUDGET_OURS, c729 + ["44444448"], idx)
    lines.append("729 configurations feasible: Plan D budget at 100 MHz %d; Plan D budget, clock classes %d; "
                 "our budget (LUT 80 %%), clock classes %d" % (len(e100), len(ecls), sum(c in ours for c in c729)))

    with open(os.path.join(args.out, "rq2_729_energies.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["config8", "g1_w", "g1_a", "g2_w", "g2_a", "g3_w", "g3_a", "model_size_bits",
                     "BOPs_input_act", "BOPs_group_act", "E_nominal_mJ_f100_planD",
                     "E_nominal_mJ_best_class_planD", "f_MHz_best_class_planD"])
        for c in c729:
            b = [int(x) for x in c]
            wr.writerow([c] + b[:6] + [es.wmem_bits(b), es.bops(b), bops_group_a(b),
                                       round(e100[c][0], 5), round(ecls[c][0], 5), ecls[c][1]])

    B = np.array([es.bops([int(x) for x in c]) for c in c729], dtype=float)
    E1 = np.array([e100[c][0] for c in c729])
    E2 = np.array([ecls[c][0] for c in c729])
    iu = np.triu_indices(len(c729), 1)

    def reversed_share(e):
        db, de = (B[:, None] - B[None, :])[iu], (e[:, None] - e[None, :])[iu]
        return 100.0 * np.mean(db * de < 0)
    lines += ["", "RQ2 on the 729 configurations (Plan D budget):",
              "  Spearman(BOPs, energy): fixed 100 MHz %.3f | best clock class %.3f"
              % (spearmanr(B, E1).correlation, spearmanr(B, E2).correlation),
              "  pairs ranked the other way by BOPs (all %d pairs): fixed 100 MHz %.1f %% | clock classes %.1f %%"
              % (len(iu[0]), reversed_share(E1), reversed_share(E2))]
    gain = 100.0 * (1 - E2 / E1)
    lines.append("  energy saved by the best clock class vs fixed 100 MHz: median %.1f %%, range %.1f to %.1f %%"
                 % (np.median(gain), gain.min(), gain.max()))

    lines += ["", "FC precision (fixed 100 MHz, Plan D budget):"]
    fc = best_per_config(args.points_dir, (100.0,), BUDGET_PLAN_D, ["44444448", "44444488"], idx)
    for c in ("44444448", "44444488"):
        e, f, mode, lut, bram, dsp = fc[c]
        lines.append("  %s %.4f mJ (%s, %d LUT, %d BRAM18, %d DSP)" % (c, e, mode, lut, bram, dsp))
    e, f, mode, lut, bram, dsp = ours["88888888"]
    lines += ["", "88888888 with our settings (clock classes, LUT 80 %%): %.4f mJ at %g MHz, %s, %d LUT, %d BRAM18, %d DSP"
              % (e, f, mode, lut, bram, dsp)]

    lines += ["", "Uniform configurations: model size (bits), BOPs (input activations; stem input 8-bit), "
              "BOPs (each layer's own group A)"]
    for c in ("88888888", "44444488", "22222288"):
        b = [int(x) for x in c]
        lines.append("  %s %d | %d | %d" % (c, es.wmem_bits(b), es.bops(b), bops_group_a(b)))

    lines += ["", "BRAM ladder, s3c2 (36,864 weights), P = PE x SIMD: FINN v0.10.1 bram_estimation vs Plan D 5.4 (4)"]
    for w in (2, 4, 8):
        lines.append("  W%d: " % w + " | ".join("P=%d FINN %d Plan D %d" % (p, finn_bram(36864, p, w), plan_d_bram(36864, p, w))
                                               for p in (16, 64, 256)))

    lines += ["", "RTL threshold nodes by output width (results/audit_dataset/rtl_nodes.csv):"]
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "audit_dataset", "rtl_nodes.csv")))
            if "Threshold" in r["op"]]
    for odt in sorted({r["outputDataType"] for r in rows}):
        rs = [r for r in rows if r["outputDataType"] == odt]
        tot = lambda k: sum(float(r[k] or 0) for r in rs)  # noqa: E731
        lines.append("  %s: %d nodes | FINN %.0f LUT, %.0f BRAM18 | Vivado %.0f LUT, %.0f BRAM18"
                     % (odt, len(rs), tot("finn_LUT"), tot("finn_BRAM_18K"), tot("vivado_LUT"), tot("vivado_BRAM18")))
    ex = [r for r in rows if r["NumChannels"] == "64" and "UINT8" in r["outputDataType"]][0]
    lines.append("  example, 64 channels, UINT8 (%s): FINN %s LUT, %s BRAM18 | Vivado %s LUT, %s BRAM18"
                 % (ex["case"], ex["finn_LUT"], ex["finn_BRAM_18K"], ex["vivado_LUT"], ex["vivado_BRAM18"]))

    m = json.load(open(os.path.join(ROOT, "results", "accuracy", "acc_model_3rep.json")))
    lines += ["", "Sensitivity, W4A4 base %.2f %%, 3 seeds: mean change (max - min over seeds)" % m["base_acc"]]
    for g in ("g1", "g2", "g3", "g4"):
        lines.append("  %s: " % g + " | ".join("%s %+.2f (%.2f)" % (v, m["delta"][g + "_" + v], m["delta_spread"][g + "_" + v])
                                               for v in ("w2a4", "w8a4", "w4a8", "w4a2")))
    out = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "plan_d_checks.txt"), "w", encoding="utf-8").write(out)
    print(out)


if __name__ == "__main__":
    main()
