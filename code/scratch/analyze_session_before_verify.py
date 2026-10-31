"""Turn a measurement session (run_session.py) into power-model parameters and the H2' test.

  python analyze_session.py SESSION_DIR [--meter-csv log.csv --time-col t --power-col p]
         [--volt-col v --amp-col a] [--util util.json]

Power per window: mean of the meter samples between pc_t0 + guard and pc_t1 - guard (PC
clock), or the power_w typed in during the session. Then, per bitstream:
  * idle (clock running) P(f) = a + b*f;  idle_halt = P_halt;  idle_slow
  * continuous P(f) = a' + b'*f and images/s, so energy per image = P / throughput
  * fixed rate R: energy per image = mean power / achieved images per second, for the
    clock running at each f, race-to-halt (rate_halt) and slowest-divider idle (rate_slow)
With --util (bitstream -> U, e.g. from FINN estimates) the idle slopes b = k_f + k_fU*U
and the active slopes b' - b = k_act*U are fitted across bitstreams.
All outputs are measurements of finn-examples CNV bitstreams, not of ResNet-8.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np


def read_meter(path, tcol, pcol, vcol, acol):
    """CSV with a header, or whitespace-separated text with a header line (e.g. the
    fnirsi_logger.py output: timestamp sample_in_packet voltage_V current_A ...)."""
    t, p = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip()]
    if "," in lines[0]:
        rows = csv.DictReader(lines)
    else:
        head = lines[0].split()
        rows = (dict(zip(head, ln.split())) for ln in lines[1:] if not ln.startswith(head[0]))
    for r in rows:
        try:
            tv = r[tcol]
            ts = float(tv) if tv.replace(".", "", 1).isdigit() else \
                datetime.fromisoformat(tv).timestamp()
            pw = float(r[pcol]) if pcol else float(r[vcol]) * float(r[acol])
        except (KeyError, ValueError):
            continue
        t.append(ts)
        p.append(pw)
    o = np.argsort(t)
    return np.array(t)[o], np.array(p)[o]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def linfit(x, y):
    if len(set(x)) < 2:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = np.array(y) - (a + b * np.array(x))
    return {"a": round(float(a), 4), "b_w_per_mhz": round(float(b), 6),
            "rms_resid_w": round(float(np.sqrt(np.mean(resid ** 2))), 4), "n": len(x)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--meter-csv")
    p.add_argument("--time-col", default="t")
    p.add_argument("--power-col", default="p")
    p.add_argument("--volt-col")
    p.add_argument("--amp-col")
    p.add_argument("--guard", type=float, default=1.0, help="seconds dropped at both ends")
    p.add_argument("--util", help="json {model: U} for the cross-bitstream fit")
    p.add_argument("--fe-est", help="dir with <model>/calibration.json from calib_cnv_mlperf.py "
                   "(FINN estimates of the same bitstreams): compares measured throughput "
                   "with f / max_cycles")
    args = p.parse_args()
    rows = list(csv.DictReader(open(os.path.join(args.session, "windows.csv"))))
    if args.meter_csv:
        mt, mp = read_meter(args.meter_csv, args.time_col,
                            None if args.volt_col else args.power_col, args.volt_col, args.amp_col)
    for r in rows:
        r["P"] = fnum(r.get("power_w"))
        if args.meter_csv:
            t0, t1 = float(r["pc_t0"]) + args.guard, float(r["pc_t1"]) - args.guard
            m = (mt >= t0) & (mt <= t1)
            r["P"] = float(mp[m].mean()) if m.sum() >= 3 else None
            r["n_samples"] = int(m.sum())
        r["dur"] = float(r["t1"]) - float(r["t0"]) if r.get("t1") else None
    ok = [r for r in rows if r["P"] is not None and r.get("ok") in ("True", "true", True)]
    by = defaultdict(list)
    for r in ok:
        key = (r["model"], r["state"], round(fnum(r["fclk0_mhz"]) or 0, 2),
               round(fnum(r["run_mhz"]) or 0, 2), fnum(r["rate"]))
        by[key].append(r)
    med = {}
    for key, rs in by.items():
        P = float(np.median([r["P"] for r in rs]))
        ips = float(np.median([fnum(r["images_per_s"]) or 0 for r in rs]))
        med[key] = {"P_w": round(P, 4), "images_per_s": round(ips, 2), "reps": len(rs),
                    "E_mJ_per_image": round(1e3 * P / ips, 4) if ips > 0 else None,
                    "late_frac": round(float(np.median([(fnum(r["late_frames"]) or 0) /
                                                       max(fnum(r["images"]) or 1, 1)
                                                       for r in rs])), 3),
                    "outputs_match": all(r.get("outputs_match") in ("True", "", None)
                                         for r in rs)}
    out = {"models": {}}
    for m in sorted({k[0] for k in med}):
        g = {k: v for k, v in med.items() if k[0] == m}
        idle = sorted((k[2], v["P_w"]) for k, v in g.items() if k[1] == "idle")
        cont = sorted((k[2], v["P_w"], v["images_per_s"]) for k, v in g.items()
                      if k[1] == "cont")
        halt = [v["P_w"] for k, v in g.items() if k[1] == "idle_halt"]
        slow = [v["P_w"] for k, v in g.items() if k[1] == "idle_slow"]
        res = {"idle_fit": linfit([x for x, _ in idle], [y for _, y in idle]),
               "cont_fit": linfit([x for x, _, _ in cont], [y for _, y, _ in cont]),
               "P_idle_halt_w": round(float(np.median(halt)), 4) if halt else None,
               "P_idle_slow_w": round(float(np.median(slow)), 4) if slow else None,
               "idle": {f: P for f, P in idle},
               "continuous": {f: {"P_w": P, "images_per_s": t,
                                  "E_mJ": round(1e3 * P / t, 4) if t else None}
                              for f, P, t in cont}}
        if res["idle_fit"] and res["P_idle_halt_w"] is not None:
            # clock-tree + idle flip-flop power at each f = idle(f) - halted idle
            res["clock_power_w"] = {f: round(P - res["P_idle_halt_w"], 4) for f, P in idle}
        fixed = defaultdict(dict)
        for k, v in g.items():
            if k[1].startswith("rate"):
                label = k[1] if k[1] != "rate" else f"rate@{k[2]:g}MHz"
                fixed[f"{k[4]:g}"][label] = {"E_mJ_per_image": v["E_mJ_per_image"],
                                             "P_w": v["P_w"], "images_per_s": v["images_per_s"],
                                             "late_frac": v["late_frac"]}
        res["fixed_rate"] = dict(fixed)
        single = sorted((k[2], v["P_w"], v["images_per_s"]) for k, v in g.items()
                        if k[1] == "single")
        res["single_stream"] = {f: {"P_w": P, "latency_ms": round(1e3 / t, 4) if t else None,
                                    "E_mJ": round(1e3 * P / t, 4) if t else None}
                                for f, P, t in single}
        cal = args.fe_est and os.path.join(args.fe_est, m, "calibration.json")
        if cal and os.path.exists(cal):
            est = json.load(open(cal))["estimate_finn_v0101"]
            res["throughput_vs_finn_estimate"] = {
                f: {"measured": t, "estimated": round(f * 1e6 / est["max_cycles"], 1),
                    "ratio": round(t / (f * 1e6 / est["max_cycles"]), 3)}
                for f, _, t in cont if t}
            if "critical_path_cycles" in est:
                res["latency_vs_finn_estimate"] = {
                    f: {"measured_ms": round(1e3 / t, 4),
                        "estimated_pl_ms": round(est["critical_path_cycles"] / (f * 1e3), 4)}
                    for f, _, t in single if t}
        out["models"][m] = res
    if args.util:
        U = json.load(open(args.util))
        ms = [m for m in out["models"] if m in U and out["models"][m]["idle_fit"]
              and out["models"][m]["cont_fit"]]
        if len(ms) >= 2:
            u = [U[m] for m in ms]
            b_idle = [out["models"][m]["idle_fit"]["b_w_per_mhz"] for m in ms]
            b_act = [out["models"][m]["cont_fit"]["b_w_per_mhz"] - bi for m, bi in zip(ms, b_idle)]
            kf = linfit(u, b_idle)
            out["global_fit"] = {"models": ms, "U": u,
                                 "k_f": kf and kf["a"], "k_fU": kf and kf["b_w_per_mhz"],
                                 "k_act_per_U": round(float(np.mean([b / x for b, x in
                                                                     zip(b_act, u)])), 6),
                                 "P0_halt_mean": round(float(np.mean(
                                     [out["models"][m]["P_idle_halt_w"] or np.nan for m in ms])), 4)}
    json.dump(out, open(os.path.join(args.session, "summary.json"), "w"), indent=1)
    gf = out.get("global_fit")
    if gf and None not in (gf["k_f"], gf["k_fU"]):
        # same form as energy_search.POWER; load with energy_search.py --power-json
        json.dump({"z2_measured": {"P0": gf["P0_halt_mean"], "k_f": gf["k_f"],
                                   "k_fU": gf["k_fU"], "k_act": gf["k_act_per_U"]}},
                  open(os.path.join(args.session, "power_params.json"), "w"), indent=1)
    lines = ["| model | idle a (W) | idle slope (mW/MHz) | halted idle (W) | slowest-clock idle (W) "
             "| cont slope (mW/MHz) |", "|---|---|---|---|---|---|"]
    for m, r in out["models"].items():
        fi, fc = r["idle_fit"] or {}, r["cont_fit"] or {}
        lines.append(f"| {m} | {fi.get('a')} | {1e3 * fi.get('b_w_per_mhz', float('nan')):.2f} | "
                     f"{r['P_idle_halt_w']} | {r['P_idle_slow_w']} | "
                     f"{1e3 * fc.get('b_w_per_mhz', float('nan')):.2f} |")
    lines.append("")
    for m, r in out["models"].items():
        for R, d in r["fixed_rate"].items():
            lines.append(f"- {m} R={R}: " + ", ".join(
                f"{k} {v['E_mJ_per_image']} mJ (late {v['late_frac']})" for k, v in sorted(d.items())))
    if "global_fit" in out:
        lines.append("\nglobal fit: " + json.dumps(out["global_fit"]))
    open(os.path.join(args.session, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    plot(out, args.session)


def plot(out, sdir):
    """power_vs_clock.png and fixed_rate_energy.png in the session folder (measured)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = list(out["models"])
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        r = out["models"][m]
        f = sorted(float(k) for k in r["idle"])
        ax.plot(f, [r["idle"][k] for k in sorted(r["idle"], key=float)], "o-", label="idle, clock running")
        cf = sorted(r["continuous"], key=float)
        ax.plot([float(k) for k in cf], [r["continuous"][k]["P_w"] for k in cf], "s-", label="continuous")
        if r["P_idle_halt_w"] is not None:
            ax.axhline(r["P_idle_halt_w"], c="g", ls="--", label="idle, FCLK0 stopped")
        if r["P_idle_slow_w"] is not None:
            ax.axhline(r["P_idle_slow_w"], c="orange", ls=":", label="idle, slowest divider")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("FCLK0 (MHz)")
    np.atleast_1d(axes)[0].set_ylabel("board power (W, measured)")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(sdir, "power_vs_clock.png"), dpi=200)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), sharey=False)
    for ax, m in zip(np.atleast_1d(axes), ms):
        fr = out["models"][m]["fixed_rate"]
        labels = sorted({k for d in fr.values() for k in d})
        x = np.arange(len(fr))
        w = 0.8 / max(len(labels), 1)
        for i, lab in enumerate(labels):
            ax.bar(x + i * w, [fr[R].get(lab, {}).get("E_mJ_per_image") or 0 for R in fr], w, label=lab)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([f"{R} FPS" for R in fr])
        ax.set_title(m, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("energy per image (mJ, measured)")
    np.atleast_1d(axes)[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(sdir, "fixed_rate_energy.png"), dpi=200)


if __name__ == "__main__":
    main()
