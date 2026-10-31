import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()


def rep(old, new, count=-1):
    global t
    assert old in t, old[:80]
    t = t.replace(old, new, count)


rep('''  fit       refit (a) RTL MVU LUT = k*hi_block + b and (b) HLS MVU compute LUT = k*(consistent -
            weight LUTRAM) on the per-unit audits plus the designs' units, (c) the SWG ratio, (d)
            a constant per small operator; leave-one-design-out: refit without a design, predict
            its whole LUT total, compare with its synthesis''',
    '''Unit models, fitted on the per-unit audits plus the designs' units:
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
            LUT total, compare with its synthesis''')

rep('''def design_nodes(designs_dirs, tables):''',
    '''def annotate(table):
    """Layer shape of every MVU node in the tables: MW, MH, input bits A, weight bits W."""
    from replay_bram import TOPOLOGY
    for key, rec in table.items():
        _, _, g, w, a, prev = key.split("|")
        nb = {"a": int(a), "prev": int(prev)}
        mvus = [n for n in rec["nodes"] if n["op"].startswith("MVAU")]
        for n, (mw, mh, src) in zip(mvus, TOPOLOGY[int(g)]):
            n["shape"] = {"MW": mw, "MH": mh, "A": nb[src], "W": int(w)}
    return table


def design_nodes(designs_dirs, tables):''', 1)

rep('''    table, _ = load_tables(tables)
    budget''', '''    table = annotate(load_tables(tables)[0])
    budget''')

rep('''    from calib_mvau_hls import load_points
    hls = load_points([f"{scratch}/audit_hls_finn", f"{scratch}/audit_hls"])
    return ([(int(c["attrs"]["PE"]) * int(c["attrs"]["SIMD"]), c["vivado"]["LUT"]) for c in rtl],
            [(p[0], p[1]) for p in hls],          # (consistent, synth) of "auto", no weight LUTRAM
            [(c["finn_estimate"]["LUT"], c["vivado"]["LUT"]) for c in swg])''',
    '''    hls, seen = [], set()
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
    return float(lanes_ * s["W"] * -(-(s["MW"] * s["MH"] // lanes_) // 64))''')

rep('''    if n["op"] == "MVAU_rtl":
        auto_hi = n["node"]["auto"].get("LUT_hi", o.get("LUT_hi", 0.0))
        lutram = o.get("LUT_hi", 0.0) - auto_hi if o is not n["node"]["auto"] else 0.0
        return o.get("LUT_hi", 0.0) - lutram, lutram''',
    '''    if n["op"] == "MVAU_rtl":
        return lanes(n["node"]), weight_lutram(n["node"], o)''')

rep('''    hls = hls_a + [(split(n)[0], n["synth"]) for n in dn
                   if n["op"] == "MVAU_hls" and not n["dsp_mult"] and "LUT_cons" in n["opt"]]
    x, y = np.array([h for h, _ in hls]), np.array([s for _, s in hls])
    k_hls = float(x @ y / (x @ x))''',
    '''    hls = hls_a + [(node_hls_features(n), n["synth"]) for n in dn
                   if n["op"] == "MVAU_hls" and not n["dsp_mult"] and "LUT_cons" in n["opt"]]
    c_hls = np.linalg.lstsq(np.array([x for x, _ in hls], float), np.array([s for _, s in hls], float),
                            rcond=None)[0]''')

rep('''    return {"k_rtl": float(k_rtl), "b_rtl": float(b_rtl), "k_hls": k_hls, "f_swg": float(f_swg), "small": small,''',
    '''    return {"k_rtl": float(k_rtl), "b_rtl": float(b_rtl), "c_hls": [float(c) for c in c_hls],
            "f_swg": float(f_swg), "small": small,''')

rep('''    if op == "MVAU_rtl":
        _, lutram = split(n)
        return f["k_rtl"] * lanes(n["node"]) + f["b_rtl"] + lutram + (BLOCK_WSTRM if lutram == 0 else 0.0)
    if op == "MVAU_hls":
        if n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            return n["pred"]
        h, wl = split(n)
        return f["k_hls"] * h + wl + (BLOCK_WSTRM if wl == 0 else 0.0)''',
    '''    if op == "MVAU_rtl":
        h, lutram = split(n)
        return f["k_rtl"] * h + f["b_rtl"] + lutram + (BLOCK_WSTRM if lutram == 0 else 0.0)
    if op == "MVAU_hls":
        if n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            return n["pred"]
        wl = split(n)[1]
        return float(np.dot(f["c_hls"], node_hls_features(n))) + wl + (BLOCK_WSTRM if wl == 0 else 0.0)''')

open(p, "w", encoding="utf-8").write(t)
print("patched")
