"""Make the threshold costs of a cost table consistent with what FINN v0.10.1 generates.

Two FINN estimates disagree with the hardware they describe (implementation_notes.md 12.27):

  MVAU_hls with thresholds inside (lut, lut_finn modes): the threshold array
  [PE][TMEM][2^B-1] is fully partitioned along PE and the thresholds, so every partition is
  its own ROM. lut_estimation adds the threshold LUTRAM and the 2^B-1 comparators only when
  ram_style_thresholds = "distributed"; with "block" (HLS forces ROM_2P_BRAM) or the default
  "auto" both terms are 0, and bram_estimation never counts thresholds. So a block/auto
  threshold memory and its comparators are free. Here the block variants are dropped and
  the auto estimate gets the distributed-threshold terms (the consistent FINN estimate;
  HLS would put such small partitions in LUTRAM anyway).

  Thresholding_rtl (dsp modes): the estimate puts every stage memory (width = threshold
  bits, depth = C/PE * 2^x for stage x) in LUTRAM, while the RTL leaves RAM_STYLE "auto"
  and the synthesizer maps deep stages to BRAM. FINN's own knob for this, depth_trigger_bram,
  forces deep stages into BRAM in the RTL, but its estimate is broken in v0.10.1 (a BRAM
  trigger alone gives URAM; with a URAM trigger too the BRAM18 count is dropped). Here each
  node gets one variant per trigger D (stages at least D deep in BRAM18, counted with FINN's
  own memutil over the BRAM18 shapes, the rest in LUTRAM), next to FINN's all-LUTRAM one.
  The all-LUTRAM variant replaces FINN's estimate, which also drops any stage whose
  min-waste primitive is a BRAM18 (a tie at 9/18/36-bit widths, e.g. the UINT9 -> UINT8
  thresholds after a residual add). The comparators are still not counted (a few hundred
  LUTs per node, see the audit).

  python threshold_consistent.py --tables cost/tables_corr_hi.json --out cost/tables_thr_hi.json
"""

import argparse
import json

TRIGGERS = (128, 256, 512, 1024, 2048, 4096, 8192)


# channels of every Thresholding_rtl node of a group (ResNet-8 stacks of 16, 32, 64 channels;
# node names are not stable across bit widths, the group is)
CHANNELS = {0: 16, 1: 32, 2: 64}


def stage_depths(C, pe, B):
    return [C // pe * 2 ** x for x in range(B)]


def finn_lut(C, pe, B, wbits):
    """FINN v0.10.1's Thresholding_rtl LUT estimate: per stage the min-waste primitive, where
    a BRAM18 tie (9/18/36-bit stages) wins and is then lost (keyed "BRAM18", read as "BRAM")."""
    from finn.util.basic import get_memutil_alternatives
    lut = 0
    for d in stage_depths(C, pe, B):
        name, (cnt, _, _) = get_memutil_alternatives((wbits, d))[0]
        lut += pe * cnt if name == "LUTRAM" else 0
    return lut


def solve_geometry(lut, C, pe, a_bits, after_add):
    """Threshold bits that reproduce FINN's estimate (output bits = the group's A).

    9-, 18- and 36-bit stages lose the same deep stages, so they can give the same estimate;
    then the post-add thresholds (the group's last) take A+1 bits (UINT(A+1) after the
    add) and the others the 18-bit accumulator (ResNet-8 accumulators here are >= 12 bits)."""
    hits = [wb for wb in range(2, 41) if finn_lut(C, pe, a_bits, wb) == lut]
    if len(hits) > 1:
        hits = [a_bits + 1] if after_add and a_bits + 1 in hits else [h for h in hits if h == 18]
    if len(hits) != 1:
        raise ValueError(f"LUT {lut}, C {C}, PE {pe}, B {a_bits}: threshold bits {hits}")
    return a_bits, hits[0]


def trigger_variants(C, pe, B, wbits):
    from finn.util.basic import get_memutil_alternatives, mem_primitives_versal
    bram_prims = {k: v for k, v in mem_primitives_versal.items() if k.startswith("BRAM")}
    out, seen = [], set()
    for D in (None,) + TRIGGERS:
        lut = bram = 0
        for d in stage_depths(C, pe, B):
            if D is not None and d >= D:
                bram += pe * get_memutil_alternatives((wbits, d), bram_prims)[0][1][0]
            else:
                lut += pe * -(-d // 64) * wbits
        if (lut, bram) not in seen:
            seen.add((lut, bram))
            out.append({"attrs": {"depth_trigger_bram": D or 0}, "LUT": float(lut),
                        "BRAM_18K": float(bram), "URAM": 0.0, "DSP": 0.0})
    return out


def fix_mvau_hls(n):
    """Drop block-threshold variants; add the distributed-threshold terms to auto."""
    vs = n.get("variants", [])
    if not vs or "ram_style_thresholds" not in vs[0]["attrs"]:
        return 0.0
    key = lambda v: (v["attrs"]["resType"], v["attrs"]["ram_style"])  # noqa: E731
    dist = {key(v): v for v in vs if v["attrs"]["ram_style_thresholds"] == "distributed"}
    deltas = [dist[key(v)]["LUT"] - v["LUT"] for v in vs
              if v["attrs"]["ram_style_thresholds"] == "block"]
    # the threshold terms do not depend on the other choices (up to FINN's int() rounding)
    assert max(deltas) - min(deltas) <= 1, deltas
    delta = max(deltas)
    n["variants"] = list(dist.values())
    if not any(n["auto"]["LUT"] == v["LUT"] and n["auto"]["BRAM_18K"] == v["BRAM_18K"]
               for v in n["variants"]):
        n["auto"]["LUT"] += delta
    return delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = json.load(open(args.tables))
    stats = {"mvau_hls_nodes": 0, "mvau_hls_delta_max": 0.0, "thr_nodes": 0}
    for k, rec in d["table"].items():
        mode, T, g, w, a, prev = k.split("|")
        thr = [i for i, n in enumerate(rec["nodes"]) if n["op"] == "Thresholding_rtl"]
        for i, n in enumerate(rec["nodes"]):
            if n["op"] == "MVAU_hls":
                delta = fix_mvau_hls(n)
                stats["mvau_hls_nodes"] += delta > 0
                stats["mvau_hls_delta_max"] = max(stats["mvau_hls_delta_max"], delta)
            elif n["op"] == "Thresholding_rtl":
                pe = n["fold"]["PE"]
                B, wbits = solve_geometry(n["auto"]["LUT"], CHANNELS[int(g)], pe, int(a),
                                          i == thr[-1])
                n["thr_geometry"] = {"C": CHANNELS[int(g)], "PE": pe, "B": B, "wbits": wbits}
                n["variants"] = trigger_variants(CHANNELS[int(g)], pe, B, wbits)
                stats["thr_nodes"] += 1
                stats["thr_lost_stages"] = stats.get("thr_lost_stages", 0) + (
                    n["variants"][0]["LUT"] != n["auto"]["LUT"])
                n["auto"]["LUT"] = n["variants"][0]["LUT"]  # all stages in LUTRAM
    d["threshold_consistent"] = True
    json.dump(d, open(args.out, "w"))
    print(stats)


if __name__ == "__main__":
    main()
