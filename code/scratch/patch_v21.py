import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()


def rep(old, new, count=-1):
    global t
    assert old in t, old[:90]
    t = t.replace(old, new, count)


# docstring: the HLS model and the override
rep('''  HLS MVU   LUT = k * (consistent - weight LUTRAM) + c * SIMD * A [SF > 1 and NF > 1]
                  + d * PE * SIMD * W [resType lut]
            the second term is the input buffer the HLS MVU keeps to replay a vector across
            output folds (FINN has no term for it); the third corrects FINN's multiplier term,
            which grows too slowly with the weight width''',
    '''  HLS MVU   LUT = k * (consistent - weight LUTRAM) + c * SIMD * A [SF > 1 and NF > 1]
                  + d * PE * SIMD * W [resType lut] + e * PE * (2^B - 1) [thresholds, NF = 1]
            the second term is the input buffer the HLS MVU keeps to replay a vector across
            output folds (FINN has no term for it); the third corrects FINN's multiplier term,
            which grows too slowly with the weight width; the fourth (e < 0) removes most of
            FINN's threshold memory and comparator cost when every PE holds one threshold set
            (NF = MH / PE = 1): the thresholds are constants and synthesis folds them into
            the comparators (FINN's default folding: PE 16, SIMD 9)
            --override: audited configurations take their synthesized LUTs (v1's lookup)''')

# annotate: output bits and whether the node has internal thresholds
rep('''            n["shape"] = {"MW": mw, "MH": mh, "A": nb[src], "W": int(w)}''',
    '''            thr = any("ram_style_thresholds" in v.get("attrs", {}) for v in n.get("variants", []))
            n["shape"] = {"MW": mw, "MH": mh, "A": nb[src], "W": int(w), "B": int(a), "thr": thr}''')

# audit points: B and thresholds
rep('''            shape = {"MW": int(at["MW"]), "MH": int(at["MH"]), "A": bits(at["inputDataType"]),
                     "W": bits(at["weightDataType"])}''',
    '''            shape = {"MW": int(at["MW"]), "MH": int(at["MH"]), "A": bits(at["inputDataType"]),
                     "W": bits(at["weightDataType"]), "B": bits(at["outputDataType"]), "thr": True}''')

rep('''def hls_features(cons, shape, fold, res_type):
    """[consistent compute estimate, buffered input bits, lane-weight bits of LUT multipliers]."""
    pe, simd = fold["PE"], fold["SIMD"]
    replay = shape["MW"] > simd and shape["MH"] > pe
    return [cons, simd * shape["A"] * replay, pe * simd * shape["W"] * (res_type == "lut")]''',
    '''def hls_features(cons, shape, fold, res_type):
    """[consistent compute estimate, buffered input bits, lane-weight bits of LUT multipliers,
    threshold steps of PEs that hold a single threshold set]."""
    pe, simd = fold["PE"], fold["SIMD"]
    replay = shape["MW"] > simd and shape["MH"] > pe
    const_thr = shape["thr"] and shape["MH"] == pe
    return [cons, simd * shape["A"] * replay, pe * simd * shape["W"] * (res_type == "lut"),
            pe * (2 ** shape["B"] - 1) * const_thr]''')

# LODO: skip designs with DSP multipliers (not comparable)
rep('''    for d in sorted({n["design"] for n in dn}):
        rest = [n for n in dn if n["design"] != d]''',
    '''    for d in sorted({n["design"] for n in dn}):
        if any(n["dsp_mult"] for n in dn if n["design"] == d):
            print(f"| {d} | (HLS MVUs with DSP multipliers: not comparable) | | | | |")
            continue
        rest = [n for n in dn if n["design"] != d]''')

# tables: override from the audits
rep('''    d = json.load(open(args.tables))
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
                    n_set += 1''',
    '''    d = json.load(open(args.tables))
    annotate(d["table"])
    synth = audited(args.scratch) if args.override else {}
    n_set = n_skip = n_over = 0
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
                    s = synth.get(audit_key(node, o)) if op == "MVAU_hls" else None
                    if s is not None:
                        wl = split(fake)[1]
                        o["LUT"] = s + wl + (BLOCK_WSTRM if wl == 0 else 0.0)
                        n_over += 1''')
rep('''    print(f"wrote {args.out}: {n_set} MVU entries ({n_skip} MVAU_hls entries without a consistent "
          f"estimate kept); fit {json.dumps(f)}")''',
    '''    d["calib_v2_override"] = bool(args.override)
    print(f"wrote {args.out}: {n_set} MVU entries, {n_over} of them from audited syntheses ({n_skip} "
          f"MVAU_hls entries without a consistent estimate kept); fit {json.dumps(f)}")''')

# the lookup helpers
rep('''def hls_features(cons, shape, fold, res_type):''',
    '''def audited(scratch):
    """Synthesized LUTs of the audited MVAU_hls units (resType lut, weights outside), by
    (MW, MH, PE, SIMD, input, weight, output bits, threshold style)."""
    bits = lambda s: int("".join(ch for ch in s if ch.isdigit()))
    out = {}
    for a in (f"{scratch}/audit_hls_finn", f"{scratch}/audit_hls"):
        for c in json.load(open(os.path.join(a, "audit_hls.json"))):
            if (c.get("synth") or {}).get("LUT"):
                t = c["attrs"]
                style = "block" if c["style"] == "block" else "distributed"  # auto = distributed
                out[(int(t["MW"]), int(t["MH"]), int(t["PE"]), int(t["SIMD"]), bits(t["inputDataType"]),
                     bits(t["weightDataType"]), bits(t["outputDataType"]), style)] = c["synth"]["LUT"]
    return out


def audit_key(node, o):
    """Key of a table entry into audited(); None unless it is an HLS MVU with LUT multipliers."""
    a = o.get("attrs", {})
    if a.get("resType", "lut") != "lut" or not node["shape"]["thr"]:
        return None
    s, f = node["shape"], node["fold"]
    style = "block" if a.get("ram_style_thresholds") == "block" else "distributed"
    return (s["MW"], s["MH"], f["PE"], f["SIMD"], s["A"], s["W"], s["B"], style)


def hls_features(cons, shape, fold, res_type):''', 1)

rep('''    p.add_argument("--keep-rtl", action="store_true",''',
    '''    p.add_argument("--override", action="store_true",
                   help="tables: audited HLS MVU configurations take their synthesized LUTs")
    p.add_argument("--keep-rtl", action="store_true",''')

open(p, "w", encoding="utf-8").write(t)
print("patched")
