"""Predictions for the board session, written down BEFORE measuring (all model estimates).

For each finn-examples bitstream (FINN v0.10.1 estimates in est/<model>/calibration.json)
and each power parameter set of energy_search.py, at each clock the session will use:

  clock power  P_idle(f) - P_halt = (k_f + k_fU*U) * f
  idle         P_idle(f)          = P0 + (k_f + k_fU*U) * f
  continuous   P_cont(f)          = P_idle(f) + k_act * f * U
  throughput   f / max_cycles     (FINN estimate; calibration says 1.66x pessimistic latency)

U uses the FINN estimate ("est") and the CNV implementation calibration ("cal": LUT x 0.945,
BRAM x 0.735, from calib_cnv_mlperf.py against the MLPerf v0.7 post-route numbers).

  python predict_board.py [--est est] [--out predictions]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from energy_search import POWER  # noqa: E402

DEV = {"LUT": 53200, "BRAM_18K": 280, "DSP": 220}
CAL = {"LUT": 0.945, "BRAM_18K": 0.735, "DSP": 1.0}
MAX_MHZ = {"cnv-w1a1": 200.0, "cnv-w1a2": 200.0, "cnv-w2a2": 200.0, "tfc-w1a1": 200.0,
           "kwsmlp-w3a3": 100.0}
CLOCKS = (25.0, 50.0, 100.0, 200.0)


def util(e, cal=False):
    s = CAL if cal else {k: 1.0 for k in CAL}
    return (s["LUT"] * e["LUT"] / DEV["LUT"] + 0.25 * s["BRAM_18K"] * e["BRAM_18K"] / DEV["BRAM_18K"]
            + 0.25 * s["DSP"] * e["DSP"] / DEV["DSP"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--est", default=os.path.join(HERE, "est"))
    p.add_argument("--out", default=os.path.join(HERE, "predictions"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pred = {}
    for m in sorted(MAX_MHZ):
        f = os.path.join(args.est, m, "calibration.json")
        if not os.path.exists(f):
            continue
        e = json.load(open(f))["estimate_finn_v0101"]
        pm = pred[m] = {"U_est": round(util(e), 4), "U_cal": round(util(e, True), 4),
                        "max_cycles": e["max_cycles"], "by_set": {}}
        for name, q in POWER.items():
            r = pm["by_set"][name] = {}
            for uk in ("U_est", "U_cal"):
                U = pm[uk]
                r[uk] = {"idle_slope_mW_per_MHz": round(1e3 * (q["k_f"] + q["k_fU"] * U), 3),
                         "cont_slope_mW_per_MHz": round(1e3 * (q["k_f"] + (q["k_fU"] + q["k_act"]) * U), 3),
                         "at": {}}
                for fc in CLOCKS:
                    if fc > MAX_MHZ[m]:
                        continue
                    clk = (q["k_f"] + q["k_fU"] * U) * fc
                    cont = q["P0"] + clk + q["k_act"] * fc * U
                    fps = fc * 1e6 / e["max_cycles"]
                    r[uk]["at"][f"{fc:g}"] = {"clock_power_W": round(clk, 4),
                                              "P_idle_W": round(q["P0"] + clk, 4),
                                              "P_cont_W": round(cont, 4),
                                              "fps_est": round(fps, 1),
                                              "E_cont_mJ": round(1e3 * cont / fps, 4)}
    # fixed-rate windows of the session (one frame at a time; latency L = FINN critical path
    # x 0.6024, the CNV calibration), latency-aware H2' (implementation_notes.md 12.18)
    fr_lines = ["", "| model | set | R (FPS) | rate @fmax | rate_halt | rate_slow | jit "
                "| halt vs rate | jit vs halt |", "|---|---|---|---|---|---|---|---|---|"]
    for m, pm in pred.items():
        e = json.load(open(os.path.join(args.est, m, "calibration.json")))["estimate_finn_v0101"]
        II, L = e["max_cycles"], 0.6024 * e["critical_path_cycles"]
        fmax, f_slow, U = MAX_MHZ[m], 1000.0 / 3969, pm["U_cal"]
        pm["fixed_rate"] = {}
        for name, q in POWER.items():
            ck, act = q["k_f"] + q["k_fU"] * U, q["k_act"] * U * II * 1e-6
            for R in (100.0, 1000.0):
                busy = min(L / (fmax * 1e6), 1.0 / R)              # one frame in flight (s)
                f_j = max(1.05 * R * II / 1e6, f_slow)          # cannot go below 0.252 MHz
                E = {"rate": (q["P0"] + ck * fmax) / R + act,
                     "rate_halt": q["P0"] / R + ck * fmax * busy + act,
                     "rate_slow": q["P0"] / R + ck * (fmax * busy + f_slow * (1.0 / R - busy)) + act,
                     "jit": (q["P0"] + ck * f_j) / R + act if f_j <= fmax else None}
                pm["fixed_rate"].setdefault(name, {})[f"{R:g}"] = {k: v and round(1e3 * v, 4)
                                                                  for k, v in E.items()}
                fr_lines.append(
                    f"| {m} | {name} | {R:g} | {1e3 * E['rate']:.3f} | {1e3 * E['rate_halt']:.3f} | "
                    f"{1e3 * E['rate_slow']:.3f} | "
                    + ("—" if E["jit"] is None else f"{1e3 * E['jit']:.3f}") + " | "
                    f"{100 * (E['rate_halt'] / E['rate'] - 1):+.1f}% | "
                    + ("—" if E["jit"] is None else f"{100 * (E['jit'] / E['rate_halt'] - 1):+.1f}%")
                    + " |")
    # batched race-to-halt (k frames per clock start, CNV at 1000 FPS): PL energy per image
    # (board power minus halted idle) = activity + clock over (L + (k - 1) II) / k cycles,
    # i.e. a + b / k with b = clock energy of the pipeline fill (policy_map.py)
    bh_lines = ["", "| model | set | PL mJ/image, k = 1 / 2 / 4 / 8 | fill share at k = 1 "
                "| L - II (cycles) |", "|---|---|---|---|---|"]
    for m, pm in pred.items():
        if not m.startswith("cnv"):
            continue
        e = json.load(open(os.path.join(args.est, m, "calibration.json")))["estimate_finn_v0101"]
        II, L, U = e["max_cycles"], 0.6024 * e["critical_path_cycles"], pm["U_cal"]
        pm["batched_halt_1000fps"] = {}
        for name, q in POWER.items():
            ck, act = (q["k_f"] + q["k_fU"] * U) * 1e-6, q["k_act"] * U * II * 1e-6
            E = {k: act + ck * (L + (k - 1) * II) / k for k in (1, 2, 4, 8)}
            pm["batched_halt_1000fps"][name] = {str(k): round(1e3 * v, 4) for k, v in E.items()}
            bh_lines.append(f"| {m} | {name} | " + " / ".join(f"{1e3 * E[k]:.4f}" for k in E)
                            + f" | {100 * ck * (L - II) / E[1]:.0f}% | {L - II:.0f} |")
    # policy-map crossover (CNV-W1A1, 300 FPS; run_session.py adds these windows): PL energy
    # per frame of each policy as clock-on cycles per frame times the per-cycle clock energy,
    # plus the same activity term; deadlines of 1.3 and 2.0 frame periods pick the winner
    px_lines = ["", "| policy (CNV-W1A1, 300 FPS) | FCLK0 (MHz) | cycles / frame (x II) | worst latency (ms) "
                "| PL mJ/frame nominal / z2_anchor |", "|---|---|---|---|---|"]
    m = "cnv-w1a1"
    if m in pred:
        e = json.load(open(os.path.join(args.est, m, "calibration.json")))["estimate_finn_v0101"]
        II, L, U, R, fmax = e["max_cycles"], 0.6024 * e["critical_path_cycles"], pred[m]["U_cal"], 300.0, MAX_MHZ[m]
        pols = [("jit streaming", 10.417, 10.417e6 / R, L / 10.417e6),
                ("streaming, 2-period deadline clock", 22.727, 22.727e6 / R, L / 22.727e6),
                ("batched halt k = 2", fmax, (L + II) / 2, 1 / R + (L + II) / (fmax * 1e6)),
                ("streaming, 1.3-period deadline clock", 35.714, 35.714e6 / R, L / 35.714e6),
                ("one-frame halt k = 1", fmax, L, L / (fmax * 1e6))]
        pred[m]["policy_crossover_300fps"] = {}
        for name, f, cyc, lat in pols:
            Es = {}
            for pn in ("nominal", "z2_anchor"):
                q = POWER[pn]
                Es[pn] = 1e3 * ((q["k_f"] + q["k_fU"] * U) * cyc * 1e-6 + q["k_act"] * U * II * 1e-6)
            pred[m]["policy_crossover_300fps"][name] = {"MHz": f, "cycles_over_II": round(cyc / II, 2),
                                                        "latency_ms": round(1e3 * lat, 2),
                                                        **{k: round(v, 4) for k, v in Es.items()}}
            px_lines.append(f"| {name} | {f:g} | {cyc / II:.2f} | {1e3 * lat:.2f} | "
                            f"{Es['nominal']:.4f} / {Es['z2_anchor']:.4f} |")
        px_lines.append("")
        px_lines.append("Predicted: with a 4.33 ms deadline (1.3 periods) batched halt k = 2 beats the 35.7 MHz "
                        "stream; with 6.67 ms (2 periods) the 22.7 MHz stream beats k = 2; jit is the floor.")
    json.dump(pred, open(os.path.join(args.out, "predictions.json"), "w"), indent=1)
    lines = ["| model | U est / cal | set | idle slope (mW/MHz) | clock power @100 / @fmax (W) "
             "| P_cont @fmax (W) | fps @fmax (est) |", "|---|---|---|---|---|---|---|"]
    for m, pm in pred.items():
        fmax = f"{MAX_MHZ[m]:g}"
        for name, r in pm["by_set"].items():
            a, b = r["U_est"], r["U_cal"]
            lines.append(f"| {m} | {pm['U_est']} / {pm['U_cal']} | {name} | "
                         f"{a['idle_slope_mW_per_MHz']} / {b['idle_slope_mW_per_MHz']} | "
                         f"{a['at']['100']['clock_power_W']} / {a['at'][fmax]['clock_power_W']} | "
                         f"{a['at'][fmax]['P_cont_W']} | {a['at'][fmax]['fps_est']} |")
    lines += fr_lines + bh_lines + px_lines
    open(os.path.join(args.out, "predictions.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
