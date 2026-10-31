"""Second LUT calibration, after the selected designs were synthesized node by node (12.33).

Design-level synthesis (verify_designs.py) found the v1 tables (tables_calov) exact on BRAM and
DSP but 14-38% low on LUT, from three sources:
  - nodes FINN estimates at 0 (AddStreams, DuplicateStreams, DownSampler, Pool, LabelSelect,
    FMPadding) and ChannelwiseOp (34),
  - the compute LUTs of RTL and HLS MVUs at foldings outside the per-unit audits (the
    DSP-aware refold and the relaxed folding produce PE/SIMD the audits did not contain),
  - SWG, which FINN overestimates (Vivado / FINN about 0.5-0.8).
The weight streamers match the model: distributed weights cost their LUTRAM (P*Q*W*ceil(D/64),
already in the tables), block weights about 10 LUTs of logic.

  fit       refit (a) RTL MVU LUT = k*hi_block + b and (b) HLS MVU compute LUT = k*(consistent -
            weight LUTRAM) on the per-unit audits plus the designs' units, (c) the SWG ratio, (d)
            a constant per small operator; leave-one-design-out: refit without a design, predict
            its whole LUT total, compare with its synthesis
  tables    apply the fit to tables_calov -> tables_calv2

  python calib_v2.py fit --designs D:/vd [D:/vd2] --tables cost/tables_calov.json --scratch <S>
  python calib_v2.py tables --designs D:/vd [D:/vd2] --tables cost/tables_calov.json --scratch <S> \
      --out cost/tables_calv2.json
"""

import argparse
import json
import os

import numpy as np

SMALL = ("AddStreams_hls", "DuplicateStreams_hls", "DownSampler_hls", "Pool_hls", "LabelSelect_hls",
         "FMPadding_rtl", "ChannelwiseOp_hls")
BLOCK_WSTRM = 10.0  # LUTs of a block-RAM weight streamer (memstream logic; 7-13 measured)


def design_nodes(designs_dirs, tables):
    """Per design node: op, chosen option (with LUT_hi / LUT_cons), synthesized LUT of the unit
    and of its weight streamer."""
    from assemble_costs import assemble, choose_variants, load_tables, options
    from energy_search import DERATE, DEVICE, RES, UTIL_W
    table, _ = load_tables(tables)
    budget = {k: DEVICE[k] * DERATE[k] for k in RES}
    out = []
    for dd in designs_dirs:
        cases = json.load(open(os.path.join(dd, "cases.json")))
        res = json.load(open(os.path.join(dd, "results.json")))
        by_design = {}
        for c in cases:
            by_design.setdefault(c["design"], []).append(c)
        for tag, cs in by_design.items():
            bits, mode, T = tag.split("_")[0], "_".join(tag.split("_")[1:-1]), int(tag.rsplit("_", 1)[1])
            nodes = assemble(table, list(map(int, bits)), mode, T)
            if mode == "lut_finn":
                chosen = [n["auto"] for n in nodes]
            else:
                pick, _ = choose_variants(nodes, budget, UTIL_W)
                chosen = [options(n)[p] for n, p in zip(nodes, pick)]
            units = [c for c in cs if c["kind"] != "wstrm"]
            assert len(units) == len(nodes), (dd, tag)
            for c, n, o in zip(units, nodes, chosen):
                v = res.get(c["case"], {})
                w = res.get(c["case"] + "_wstrm", {})
                if v.get("returncode") != 0 or v.get("LUT") is None:
                    continue
                forced = c.get("attrs", {}).get("resType")
                out.append({"design": f"{os.path.basename(dd)}:{tag}", "case": os.path.join(dd, c["case"]),
                            "op": n["op"], "opt": o, "node": n,
                            "pred": c["predicted"]["LUT"], "synth": v["LUT"],
                            "wstrm": (w.get("LUT") or 0.0) if w else 0.0, "has_wstrm": bool(w),
                            "dsp_mult": n["op"] == "MVAU_hls" and mode == "lut_finn" and forced != "lut"})
    return out


def audit_points(scratch):
    rtl = [c for c in json.load(open(f"{scratch}/audit/audit.json")) if c["op"] == "MVAU_rtl"
           and "bracket" in c and c["vivado"]["returncode"] == 0]
    swg = [c for c in json.load(open(f"{scratch}/audit/audit.json"))
           if c["op"] == "ConvolutionInputGenerator_rtl" and c["vivado"]["returncode"] == 0]
    from calib_mvau_hls import load_points
    hls = load_points([f"{scratch}/audit_hls_finn", f"{scratch}/audit_hls"])
    return ([(int(c["attrs"]["PE"]) * int(c["attrs"]["SIMD"]), c["vivado"]["LUT"]) for c in rtl],
            [(p[0], p[1]) for p in hls],          # (consistent, synth) of "auto", no weight LUTRAM
            [(c["finn_estimate"]["LUT"], c["vivado"]["LUT"]) for c in swg])


def lanes(node):
    """PE * SIMD of an MVU node: the RTL MVU's LUTs follow its lane count (10.5 per lane), not
    FINN's HLS-style estimate of the same layer (the v1 input), whose bit-width terms describe
    LUT multipliers the DSP-based RTL core does not have."""
    return float(node["fold"]["PE"] * node["fold"]["SIMD"])


def split(n):
    """(estimator input of the unit model, weight LUTRAM) of a design node's chosen option."""
    o = n["opt"]
    if n["op"] == "MVAU_rtl":
        auto_hi = n["node"]["auto"].get("LUT_hi", o.get("LUT_hi", 0.0))
        lutram = o.get("LUT_hi", 0.0) - auto_hi if o is not n["node"]["auto"] else 0.0
        return o.get("LUT_hi", 0.0) - lutram, lutram
    if n["op"] == "MVAU_hls":
        vs = n["node"].get("variants", [])
        blk = {v["attrs"].get("resType"): v.get("LUT_cons", v["LUT"]) for v in vs
               if v.get("attrs", {}).get("ram_style") == "block"}
        cons = o.get("LUT_cons", o["LUT"])
        wl = max(0.0, cons - blk.get(o.get("attrs", {}).get("resType", "lut"), cons)) \
            if o.get("attrs", {}).get("ram_style") == "distributed" else 0.0
        return cons - wl, wl
    return o["LUT"], 0.0


def fit(dn, audits):
    rtl_a, hls_a, swg_a = audits
    rtl = rtl_a + [(lanes(n["node"]), n["synth"]) for n in dn if n["op"] == "MVAU_rtl"]
    X = np.array([[h, 1.0] for h, _ in rtl])
    k_rtl, b_rtl = np.linalg.lstsq(X, np.array([s for _, s in rtl]), rcond=None)[0]
    hls = hls_a + [(split(n)[0], n["synth"]) for n in dn
                   if n["op"] == "MVAU_hls" and not n["dsp_mult"] and "LUT_cons" in n["opt"]]
    x, y = np.array([h for h, _ in hls]), np.array([s for _, s in hls])
    k_hls = float(x @ y / (x @ x))
    swg = swg_a + [(n["opt"]["LUT"], n["synth"]) for n in dn if n["op"] == "ConvolutionInputGenerator_rtl"]
    f_swg = sum(s for _, s in swg) / sum(e for e, _ in swg)
    small = {op: float(np.mean([n["synth"] for n in dn if n["op"] == op]))
             for op in SMALL if any(n["op"] == op for n in dn)}
    return {"k_rtl": float(k_rtl), "b_rtl": float(b_rtl), "k_hls": k_hls, "f_swg": float(f_swg), "small": small,
            "n_rtl": len(rtl), "n_hls": len(hls), "n_swg": len(swg)}


def predict_node(n, f):
    op = n["op"]
    if op == "MVAU_rtl":
        _, lutram = split(n)
        return f["k_rtl"] * lanes(n["node"]) + f["b_rtl"] + lutram + (BLOCK_WSTRM if lutram == 0 else 0.0)
    if op == "MVAU_hls":
        if n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            return n["pred"]
        h, wl = split(n)
        return f["k_hls"] * h + wl + (BLOCK_WSTRM if wl == 0 else 0.0)
    if op == "ConvolutionInputGenerator_rtl":
        return f["f_swg"] * n["opt"]["LUT"]
    if op in f["small"]:
        return f["small"][op]
    return n["pred"]


def cmd_fit(args):
    dn = design_nodes(args.designs, args.tables)
    audits = audit_points(args.scratch)
    f = fit(dn, audits)
    print("fit on everything:", json.dumps(f))
    print("| design | v1 prediction | v2 (leave this design out) | synthesis (units + weight streamers) | v1 error | v2 error |")
    print("|---|---|---|---|---|---|")
    for d in sorted({n["design"] for n in dn}):
        rest = [n for n in dn if n["design"] != d]
        fd = fit(rest, audits)
        mine = [n for n in dn if n["design"] == d]
        syn = sum(n["synth"] + n["wstrm"] for n in mine)
        v1 = sum(n["pred"] for n in mine)
        v2 = sum(predict_node(n, fd) for n in mine)
        print(f"| {d} | {v1:.0f} | {v2:.0f} | {syn:.0f} | {100 * (v1 / syn - 1):+.1f}% | {100 * (v2 / syn - 1):+.1f}% |")
    json.dump(f, open(os.path.join(args.scratch, "cost", "calib_v2.json"), "w"), indent=1)


def cmd_tables(args):
    f = fit(design_nodes(args.designs, args.tables), audit_points(args.scratch))
    d = json.load(open(args.tables))
    n_set = 0
    for rec in d["table"].values():
        for node in rec["nodes"]:
            op = node["op"]
            entries = [node["auto"]] + node.get("variants", [])
            for o in entries:
                fake = {"op": op, "opt": o, "node": node, "pred": o["LUT"], "dsp_mult": False}
                if op in ("MVAU_rtl", "MVAU_hls") and ("LUT_hi" in o or "LUT_cons" in o):
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = predict_node(fake, f)
                    n_set += 1
                elif op == "ConvolutionInputGenerator_rtl":
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = f["f_swg"] * o["LUT"]
                elif op in f["small"]:
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = max(o["LUT"], f["small"][op])
    d["calib_v2"] = f
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n_set} MVU entries; fit {json.dumps(f)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "tables"])
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--tables", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    {"fit": cmd_fit, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
