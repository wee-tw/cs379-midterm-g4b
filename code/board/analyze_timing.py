"""Timing-only board session (run_session.py --plan timing; no power meter): what the hardware
does per image, against FINN v0.10.1's cycle estimates for the same finn-examples bitstreams.

Per bitstream:
  * continuous (batch 1000): time per image t(f) = a + II / f, fitted over the clocks: II is the
    measured cycles per image (the pipeline bottleneck), a the per-image software overhead
  * single image (batch 1): t(f) = a1 + L / f: L is the measured single-image latency in cycles,
    a1 the per-call driver overhead
  * L / II, the pipeline depth in initiation intervals that the clock-policy map uses (r)
  * FINN's estimates for the same bitstreams (est/<model>/calibration.json): max_cycles (II) and
    critical_path_cycles (L), and measured / estimated
  * fixed-rate policies: achieved rate and share of late frames for race-to-halt (k frames per
    clock start), the slowest-divider idle and the just-in-time clocks
  * clock throttle: probe and verify results, halts and failed halts per window
  * top-1 accuracy on 1000 CIFAR-10 test images (CNV)
All numbers are measurements of the finn-examples bitstreams on the board, not of ResNet-8.
Writes timing_summary.json and timing_summary.md into the session folder.

  python analyze_timing.py SESSION_DIR [--est est]
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fit_cycles(points):
    """points: [(f_MHz, images_per_s)]; t = a + c / f -> (c cycles, a seconds), or None."""
    pts = [(f, ips) for f, ips in points if f and ips and ips > 0]
    if len({f for f, _ in pts}) < 2:
        return None
    x = np.array([1.0 / (f * 1e6) for f, _ in pts])
    y = np.array([1.0 / ips for _, ips in pts])
    c, a = np.polyfit(x, y, 1)
    resid = y - (a + c * x)
    return {"cycles": round(float(c), 1), "overhead_us": round(1e6 * float(a), 2),
            "rms_resid_pct": round(100 * float(np.sqrt(np.mean((resid / y) ** 2))), 2), "n": len(pts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--est", default=os.path.join(HERE, "est"))
    args = ap.parse_args()
    plan = json.load(open(os.path.join(args.session, "plan.json")))
    step_model, cur = {}, None
    for i, (kind, payload) in enumerate(plan["steps"]):
        if kind == "cmd" and payload.get("cmd") == "load":
            cur = payload["model"]
        step_model[i] = cur
    events = defaultdict(dict)
    for line in open(os.path.join(args.session, "responses.jsonl")):
        r = json.loads(line)
        m = step_model.get(r.get("step"))
        cmd = plan["steps"][r["step"]][1].get("cmd") if r.get("step") is not None else None
        if m is None or cmd not in ("accuracy", "probe_throttle", "verify_throttle", "probe_slow", "load"):
            continue
        keep = {k: r.get(k) for k in ("ok", "top1", "n", "stuck", "note", "halt_verified", "stop_seq",
                                      "outputs_match", "error", "slowest_mhz") if k in r}
        events[m][cmd] = keep
    rows = [r for r in csv.DictReader(open(os.path.join(args.session, "windows.csv")))
            if r.get("ok") in ("True", "true")]
    out = {"models": {}}
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]

        def per_clock(state):
            by = defaultdict(list)
            for r in rs:
                if r["state"] == state:
                    by[round(fnum(r["fclk0_mhz"]) or 0, 3)].append(fnum(r["images_per_s"]) or 0)
            return {f: float(np.median(v)) for f, v in sorted(by.items())}
        cont, single = per_clock("cont"), per_clock("single")
        res = {"events": events.get(m, {}),
               "continuous": {f: {"images_per_s": round(v, 2), "cycles_per_image": round(f * 1e6 / v, 1)}
                              for f, v in cont.items() if v > 0},
               "single": {f: {"latency_ms": round(1e3 / v, 4), "cycles_per_call": round(f * 1e6 / v, 1)}
                          for f, v in single.items() if v > 0},
               "II_fit": fit_cycles(list(cont.items())),
               "L_fit": fit_cycles(list(single.items()))}
        cal = os.path.join(args.est, m, "calibration.json")
        if os.path.exists(cal):
            est = json.load(open(cal))["estimate_finn_v0101"]
            res["finn_estimate"] = {"II_cycles": est.get("max_cycles"), "L_cycles": est.get("critical_path_cycles")}
            if res["II_fit"] and est.get("max_cycles"):
                res["II_measured_over_estimated"] = round(res["II_fit"]["cycles"] / est["max_cycles"], 3)
            if res["L_fit"] and est.get("critical_path_cycles"):
                res["L_measured_over_estimated"] = round(res["L_fit"]["cycles"] / est["critical_path_cycles"], 3)
            if est.get("max_cycles") and est.get("critical_path_cycles"):
                res["r_estimated"] = round(est["critical_path_cycles"] / est["max_cycles"], 2)
        if res["II_fit"] and res["L_fit"] and res["II_fit"]["cycles"] > 0:
            res["r_measured"] = round(res["L_fit"]["cycles"] / res["II_fit"]["cycles"], 2)
        pol = defaultdict(list)
        for r in rs:
            if r["state"] in ("rate_halt", "rate_slow", "jit", "rate"):
                k = int(fnum(r.get("k")) or 1)
                key = f"{r['state']}{'_k%d' % k if k > 1 else ''} @ {fnum(r['rate']) or 0:g} FPS, " \
                      f"{fnum(r['fclk0_mhz']) or 0:g} MHz"
                imgs = fnum(r["images"]) or 0
                pol[key].append((fnum(r["images_per_s"]) or 0, (fnum(r["late_frames"]) or 0) / max(imgs, 1),
                                 fnum(r.get("halts")) or 0, fnum(r.get("halt_fail")) or 0))
        res["fixed_rate"] = {k: {"achieved_fps": round(float(np.median([v[0] for v in vs])), 2),
                                 "late_frac": round(float(np.median([v[1] for v in vs])), 3),
                                 "halts": int(sum(v[2] for v in vs)), "halt_fail": int(sum(v[3] for v in vs))}
                             for k, vs in sorted(pol.items())}
        out["models"][m] = res
    json.dump(out, open(os.path.join(args.session, "timing_summary.json"), "w"), indent=1)
    md = ["# Timing-only board session (finn-examples bitstreams on PYNQ-Z2; measured)", ""]
    md += ["| model | top-1 (1000 img) | throttle (probe / verified) | II measured (cycles) | FINN II | meas/est | "
           "L measured (cycles) | FINN L | meas/est | r = L/II meas / est | per-image overhead cont / single (us) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, res in out["models"].items():
        ev, fe = res["events"], res.get("finn_estimate", {})
        acc = (ev.get("accuracy") or {}).get("top1")
        thr = f"{(ev.get('probe_throttle') or {}).get('ok')} / {(ev.get('verify_throttle') or {}).get('halt_verified')}"
        ii, ll = res["II_fit"] or {}, res["L_fit"] or {}
        md.append(f"| {m} | {acc} | {thr} | {ii.get('cycles')} | {fe.get('II_cycles')} | "
                  f"{res.get('II_measured_over_estimated')} | {ll.get('cycles')} | {fe.get('L_cycles')} | "
                  f"{res.get('L_measured_over_estimated')} | {res.get('r_measured')} / {res.get('r_estimated')} | "
                  f"{ii.get('overhead_us')} / {ll.get('overhead_us')} |")
    md += ["", "## Fixed-rate policies (achieved rate, share of late frames)", ""]
    for m, res in out["models"].items():
        for k, v in res["fixed_rate"].items():
            md.append(f"- {m} {k}: {v['achieved_fps']} FPS, late {v['late_frac']}, halts {v['halts']}, "
                      f"failed halts {v['halt_fail']}")
    open(os.path.join(args.session, "timing_summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
