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

Unit models, fitted on the per-unit audits plus the designs' units:
  RTL MVU   LUT = k * PE * SIMD + b (the DSP-based core's LUTs follow its lane count; v1 scaled
            FINN's HLS-style estimate of the same layer, whose bit-width terms describe LUT
            multipliers the RTL core does not have, and missed the DSP-aware refold's units)
  HLS MVU   LUT = k * (consistent - weight LUTRAM) + c * SIMD * A [SF > 1 and NF > 1]
                  + d * PE * SIMD * W [resType lut]
            the second term is the input buffer the HLS MVU keeps to replay a vector across
            output folds (FINN has no term for it); the third corrects FINN's multiplier term,
            which grows too slowly with the weight width
  weights   distributed: P*Q*W*ceil(depth/64) LUTRAM; block: about 10 LUTs of streamer logic

  fit       the fits, then leave-one-design-out: refit without a design, predict its whole
            LUT total, compare with its synthesis
  tables    apply the fit to tables_calov -> tables_calv2

  python calib_v2.py fit --designs D:/vd [D:/vd2] --tables cost/tables_calov.json --scratch <S>
  python calib_v2.py tables --designs D:/vd [D:/vd2] --tables cost/tables_calov.json --scratch <S> \
      --out cost/tables_calv2.json
  python calib_v2.py tables --fit cost/calib_v2.json [--keep-rtl] --tables <harm table> --scratch <S> --out ...
"""

import argparse
import json
import os

import numpy as np

SMALL = ("AddStreams_hls", "DuplicateStreams_hls", "DownSampler_hls", "Pool_hls", "LabelSelect_hls",
         "FMPadding_rtl", "ChannelwiseOp_hls")
BLOCK_WSTRM = 10.0  # LUTs of a block-RAM weight streamer (memstream logic; 7-13 measured)


def annotate(table):
    """Layer shape of every MVU node in the tables: MW, MH, input bits A, weight bits W."""
    from replay_bram import TOPOLOGY
    for key, rec in table.items():
        _, _, g, w, a, prev = key.split("|")
        nb = {"a": int(a), "prev": int(prev)}
        mvus = [n for n in rec["nodes"] if n["op"].startswith("MVAU")]
        for n, (mw, mh, src) in zip(mvus, TOPOLOGY[int(g)]):
            n["shape"] = {"MW": mw, "MH": mh, "A": nb[src], "W": int(w)}
    return table


def design_nodes(designs_dirs, tables):
    """Per design node: op, chosen option (with LUT_hi / LUT_cons), synthesized LUT of the unit
    and of its weight streamer."""
    from assemble_costs import assemble, choose_variants, load_tables, options
    from energy_search import DERATE, DEVICE, RES, UTIL_W
    table = annotate(load_tables(tables)[0])
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
    hls, seen = [], set()
    bits = lambda s: int("".join(ch for ch in s if ch.isdigit()))
    for a in (f"{scratch}/audit_hls_finn", f"{scratch}/audit_hls"):
        by = {}
        for c in json.load(open(os.path.join(a, "audit_hls.json"))):
            by.setdefault(c["case"].rsplit("_", 1)[0], {})[c["style"]] = c
        for v in by.values():
            if not all((v.get(s, {}).get("synth") or {}).get("LUT") for s in ("auto", "distributed")):
                continue
            at = v["auto"]["attrs"]
            key = (tuple(sorted(at.items())), v["auto"]["synth"]["LUT"])
            if key in seen:  # the same node in two audits synthesizes identically
                continue
            seen.add(key)
            shape = {"MW": int(at["MW"]), "MH": int(at["MH"]), "A": bits(at["inputDataType"]),
                     "W": bits(at["weightDataType"])}
            fold = {"PE": int(at["PE"]), "SIMD": int(at["SIMD"])}
            # consistent estimate = FINN's "distributed" one; the audits all used resType lut
            hls.append((hls_features(v["distributed"]["finn_estimate"]["LUT"], shape, fold, "lut"),
                        v["auto"]["synth"]["LUT"]))
    return ([(int(c["attrs"]["PE"]) * int(c["attrs"]["SIMD"]), c["vivado"]["LUT"]) for c in rtl], hls,
            [(c["finn_estimate"]["LUT"], c["vivado"]["LUT"]) for c in swg])


def hls_features(cons, shape, fold, res_type):
    """[consistent compute estimate, buffered input bits, lane-weight bits of LUT multipliers]."""
    pe, simd = fold["PE"], fold["SIMD"]
    replay = shape["MW"] > simd and shape["MH"] > pe
    return [cons, simd * shape["A"] * replay, pe * simd * shape["W"] * (res_type == "lut")]


def node_hls_features(n):
    rt = n["opt"].get("attrs", {}).get("resType", "lut")
    return hls_features(split(n)[0], n["node"]["shape"], n["node"]["fold"], rt)


def weight_lutram(node, opt):
    """LUTRAM of distributed weights: width P*Q*W, one LUT per bit per 64 words."""
    if opt.get("attrs", {}).get("ram_style") != "distributed":
        return 0.0
    s, f = node["shape"], node["fold"]
    lanes_ = f["PE"] * f["SIMD"]
    return float(lanes_ * s["W"] * -(-(s["MW"] * s["MH"] // lanes_) // 64))


def lanes(node):
    """PE * SIMD of an MVU node: the RTL MVU's LUTs follow its lane count (10.5 per lane), not
    FINN's HLS-style estimate of the same layer (the v1 input), whose bit-width terms describe
    LUT multipliers the DSP-based RTL core does not have."""
    return float(node["fold"]["PE"] * node["fold"]["SIMD"])


def split(n):
    """(estimator input of the unit model, weight LUTRAM) of a design node's chosen option."""
    o = n["opt"]
    if n["op"] == "MVAU_rtl":
        return lanes(n["node"]), weight_lutram(n["node"], o)
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
    hls = hls_a + [(node_hls_features(n), n["synth"]) for n in dn
                   if n["op"] == "MVAU_hls" and not n["dsp_mult"] and "LUT_cons" in n["opt"]]
    c_hls = np.linalg.lstsq(np.array([x for x, _ in hls], float), np.array([s for _, s in hls], float),
                            rcond=None)[0]
    swg = swg_a + [(n["opt"]["LUT"], n["synth"]) for n in dn if n["op"] == "ConvolutionInputGenerator_rtl"]
    f_swg = sum(s for _, s in swg) / sum(e for e, _ in swg)
    small = {op: float(np.mean([n["synth"] for n in dn if n["op"] == op]))
             for op in SMALL if any(n["op"] == op for n in dn)}
    return {"k_rtl": float(k_rtl), "b_rtl": float(b_rtl), "c_hls": [float(c) for c in c_hls],
            "f_swg": float(f_swg), "small": small,
            "n_rtl": len(rtl), "n_hls": len(hls), "n_swg": len(swg)}


def predict_node(n, f):
    op = n["op"]
    if op == "MVAU_rtl":
        h, lutram = split(n)
        return f["k_rtl"] * h + f["b_rtl"] + lutram + (BLOCK_WSTRM if lutram == 0 else 0.0)
    if op == "MVAU_hls":
        if n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            return n["pred"]
        wl = split(n)[1]
        return float(np.dot(f["c_hls"], node_hls_features(n))) + wl + (BLOCK_WSTRM if wl == 0 else 0.0)
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
    # the fit comes from the tables the designs were selected with; --fit reuses it on other tables
    f = json.load(open(args.fit)) if args.fit else fit(design_nodes(args.designs, args.tables),
                                                        audit_points(args.scratch))
    d = json.load(open(args.tables))
    annotate(d["table"])
    n_set = n_skip = 0
    for rec in d["table"].values():
        for node in rec["nodes"]:
            op = node["op"]
            entries = [node["auto"]] + node.get("variants", [])
            for o in entries:
                fake = {"op": op, "opt": o, "node": node, "pred": o["LUT"], "dsp_mult": False}
                if (op == "MVAU_rtl" and not args.keep_rtl) or (op == "MVAU_hls" and "LUT_cons" in o):
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = predict_node(fake, f)
                    n_set += 1
                elif op == "MVAU_hls":
                    n_skip += 1
                elif op == "ConvolutionInputGenerator_rtl":
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = f["f_swg"] * o["LUT"]
                elif op in f["small"]:
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = max(o["LUT"], f["small"][op])
    d["calib_v2"] = f
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n_set} MVU entries ({n_skip} MVAU_hls entries without a consistent "
          f"estimate kept); fit {json.dumps(f)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "tables"])
    p.add_argument("--designs", nargs="+")
    p.add_argument("--tables", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--out")
    p.add_argument("--fit", help="tables: apply this saved fit (calib_v2.json) instead of refitting")
    p.add_argument("--keep-rtl", action="store_true",
                   help="tables: leave the MVAU_rtl LUTs as they are (FINN's view in the harm study)")
    args = p.parse_args()
    {"fit": cmd_fit, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
