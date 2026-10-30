"""HLS-calibrated LUT cost of FINN v0.10.1's MVAU_hls with internal thresholds (12.27, 12.31).

FINN counts an MVAU_hls's threshold storage and its 2^B - 1 comparators only for
ram_style_thresholds = "distributed"; the threshold-consistent tables (threshold_consistent.py)
charge that formula for "auto" as well. Vitis HLS 2022.2 synthesis of the audited nodes
(audit_hls_thresholds.py run, export_design -flow syn) shows that "auto" and "distributed"
synthesize to the same LUTs and no BRAM, well below the formula. So:

  fit     least squares LUT_synth(auto) = k * LUT_consistent (FINN's "distributed" estimate)
          over the distinct audited nodes, with leave-one-out error
  tables  scale the compute-and-threshold LUTs of every MVAU_hls with thresholds by k; the
          weight LUTRAM of distributed-weight variants is kept as FINN estimates it (with
          internal_decoupled weights it sits outside the HLS kernel, so it was not audited),
          and MVAU_hls without thresholds keep FINN's estimate. With --override, a node that
          was synthesized (same MW, MH, PE, SIMD and bit widths) takes its synthesized LUTs
          instead of the fit (resType lut; the dsp variants keep the fit)

  python calib_mvau_hls.py fit --audit audit_hls_finn audit_hls
  python calib_mvau_hls.py tables --audit audit_hls_finn audit_hls --tables cost/tables_viv.json \
      --out cost/tables_cal.json [--override]
"""

import argparse
import json
import os
import re

import numpy as np

from replay_bram import TOPOLOGY


def load_points(audits):
    """(consistent estimate, synthesized LUTs of "auto", output type) per distinct node; the same
    node in two audits synthesizes identically, a node with trained weights does not."""
    seen, pts = set(), []
    for a in audits:
        by = {}
        for c in json.load(open(os.path.join(a, "audit_hls.json"))):
            by.setdefault(c["case"].rsplit("_", 1)[0], {})[c["style"]] = c
        for v in by.values():
            if not all((v.get(s, {}).get("synth") or {}).get("LUT") for s in ("auto", "distributed")):
                continue
            key = (tuple(sorted(v["auto"]["attrs"].items())), v["auto"]["synth"]["LUT"])
            if key in seen:
                continue
            seen.add(key)
            pts.append((v["distributed"]["finn_estimate"]["LUT"], v["auto"]["synth"]["LUT"],
                        v["auto"]["finn_estimate"]["LUT"], v["auto"]["attrs"]["outputDataType"]))
    return pts


def fit(x, y):
    k = float(x @ y / (x @ x))
    loo = [abs((x[m] @ y[m] / (x[m] @ x[m])) * x[i] - y[i]) / y[i]
           for i in range(len(x)) for m in [np.arange(len(x)) != i]]
    return k, np.array(loo)


def cmd_fit(args):
    pts = load_points(args.audit)
    x, y, fa = (np.array([p[i] for p in pts]) for i in range(3))
    k, loo = fit(x, y)
    print(f"{len(pts)} nodes: LUT_synth = {k:.3f} * LUT_consistent; leave-one-out error median "
          f"{100 * np.median(loo):.0f}%, 90th percentile {100 * np.percentile(loo, 90):.0f}%, "
          f"worst {100 * loo.max():.0f}%")
    for t in sorted({p[3] for p in pts}):
        m = np.array([p[3] == t for p in pts])
        print(f"  {t}: {m.sum()} nodes, synth / FINN 'auto' median {np.median(y[m] / fa[m]):.2f} "
              f"({np.min(y[m] / fa[m]):.2f}-{np.max(y[m] / fa[m]):.2f}), synth / consistent median "
              f"{np.median(y[m] / x[m]):.2f} ({np.min(y[m] / x[m]):.2f}-{np.max(y[m] / x[m]):.2f})")


def synthesized(audits):
    """Synthesized LUTs of "auto" by (MW, MH, PE, SIMD, input, weight, output bits)."""
    bits = lambda s: int(re.sub(r"\D", "", s))  # noqa: E731
    out = {}
    for a in audits:
        for c in json.load(open(os.path.join(a, "audit_hls.json"))):
            if c["style"] == "auto" and (c.get("synth") or {}).get("LUT"):
                t = c["attrs"]
                out[(int(t["MW"]), int(t["MH"]), int(t["PE"]), int(t["SIMD"]), bits(t["inputDataType"]),
                     bits(t["weightDataType"]), bits(t["outputDataType"]))] = c["synth"]["LUT"]
    return out


def cmd_tables(args):
    pts = load_points(args.audit)
    k, _ = fit(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    synth = synthesized(args.audit) if args.override else {}
    d = json.load(open(args.tables))
    n_set = n_over = 0
    for key, rec in d["table"].items():
        _, _, g, w, a, prev = key.split("|")
        nb = {"a": int(a), "prev": int(prev)}
        mvus = [n for n in rec["nodes"] if n["op"].startswith("MVAU")]
        for n, (mw, mh, src) in zip(mvus, TOPOLOGY[int(g)]):
            vs = n.get("variants", [])
            if n["op"] != "MVAU_hls" or not vs or "ram_style_thresholds" not in vs[0]["attrs"]:
                continue
            s = synth.get((mw, mh, n["fold"]["PE"], n["fold"]["SIMD"], nb[src], int(w), int(a)))
            for r in [n["auto"]] + vs:
                r.setdefault("LUT_cons", r["LUT"])
            block = {v["attrs"]["resType"]: v["LUT_cons"] for v in vs if v["attrs"]["ram_style"] == "block"}
            wl = [max(0.0, v["LUT_cons"] - block[v["attrs"]["resType"]])
                  if v["attrs"]["ram_style"] == "distributed" else 0.0 for v in vs]
            # "auto" is one of the variants as FINN resolves it; take that variant's weight LUTRAM
            auto = n["auto"]
            same = [(v, x) for v, x in zip(vs, wl) if abs(v["LUT_cons"] - auto["LUT_cons"]) <= 1
                    and v["BRAM_18K"] == auto["BRAM_18K"]]
            res = [same[0][0]["attrs"]["resType"] if same else "lut"] + [v["attrs"]["resType"] for v in vs]
            for r, x, rt in zip([auto] + vs, [same[0][1] if same else 0.0] + wl, res):
                # the synthesized nodes used resType lut; the dsp variants keep the fit
                r["LUT"] = (s if s is not None and rt == "lut" else k * (r["LUT_cons"] - x)) + x
                n_set += 1
            n_over += s is not None
    d["mvau_hls_lut_calibration"] = {"k": k, "override": bool(args.override),
                                     "audits": [os.path.basename(p) for p in args.audit]}
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n_set} MVAU_hls entries set to {k:.3f} * consistent (+ weight LUTRAM)"
          + (f"; {n_over} nodes take their synthesized LUTs" if args.override else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "tables"])
    p.add_argument("--audit", nargs="+", required=True)
    p.add_argument("--tables")
    p.add_argument("--out")
    p.add_argument("--override", action="store_true")
    args = p.parse_args()
    {"fit": cmd_fit, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
