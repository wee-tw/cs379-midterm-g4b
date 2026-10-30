"""Block RAM of the RTL matrix-vector unit's replay buffer, which FINN v0.10.1 does not estimate (12.31).

mvu_vvu_axi.sv replays every input vector NF = MH/PE times through replay_buffer.sv: a memory of
2^max(1, ceil(log2 SF)) words of SIMD*A bits (SF = MW/SIMD), with no ram_style, and a plain
pass-through when NF = 1. FINN's MVAU_rtl estimate counts only the weight memory. Vivado 2022.2
(out of context, xc7z020) maps the buffer to BRAM18 from 1,152 bits up and to LUTRAM up to 768
bits in the 53 audited units (audit_rtl_ops.py vivado); the LUTRAM case is already inside the
Vivado-calibrated LUT cost (calib_mvu_vivado.py), the BRAM case is in no cost.

  check   the rule against the Vivado audit (every MVAU_rtl unit)
  tables  add the buffer's BRAM18 to every MVAU_rtl entry (auto and variants) of a cost table

  python replay_bram.py check --audit audit
  python replay_bram.py tables --tables cost/tables_cal.json --out cost/tables_cal_rb.json
"""

import argparse
import glob
import json
import math
import os
import re

# MVAU_rtl nodes of each group in graph order: (MW, MH, input width: "prev" group's A or own "a")
TOPOLOGY = {0: [(27, 16, "prev"), (144, 16, "a"), (144, 16, "a")],
            1: [(16, 32, "prev"), (144, 32, "prev"), (288, 32, "a")],
            2: [(32, 64, "prev"), (288, 64, "prev"), (576, 64, "a")],
            3: [(64, 10, "a")]}
OFM_PIXELS = {0: 1024, 1: 256, 2: 64, 3: 1}
BRAM_MIN_BITS = 1152  # smallest buffer Vivado put in BRAM; the largest in LUTRAM had 768 bits
BRAM18_SHAPES = ((512, 36), (1024, 18), (2048, 9), (4096, 4), (8192, 2), (16384, 1))


def replay_bram18(mw, mh, pe, simd, a_bits):
    sf, nf = mw // simd, mh // pe
    if nf == 1:
        return 0
    depth, width = 2 ** max(1, math.ceil(math.log2(sf))), simd * a_bits
    if depth * width < BRAM_MIN_BITS:
        return 0
    return min(math.ceil(width / w) * math.ceil(depth / d) for d, w in BRAM18_SHAPES)


def cmd_check(args):
    bad = 0
    cases = [c for c in json.load(open(os.path.join(args.audit, "audit.json"))) if c["op"] == "MVAU_rtl"]
    for c in cases:
        w = open(glob.glob(os.path.join(args.audit, c["case"], "*_wrapper.v"))[0]).read()
        p = {k: int(float(re.search(rf"parameter\s+{k}\s*=\s*([\d.]+)", w).group(1)))
             for k in ("MW", "MH", "PE", "SIMD", "ACTIVATION_WIDTH")}
        pred = replay_bram18(p["MW"], p["MH"], p["PE"], p["SIMD"], p["ACTIVATION_WIDTH"])
        if pred != c["vivado"]["BRAM18"]:
            bad += 1
            print("mismatch", c["case"], p, "rule", pred, "vivado", c["vivado"]["BRAM18"])
    print(f"{len(cases) - bad} of {len(cases)} MVAU_rtl units match the rule")


def cmd_tables(args):
    d = json.load(open(args.tables))
    n_set, n_bram = 0, 0
    for key, rec in d["table"].items():
        _, _, g, _, a, prev = key.split("|")
        g = int(g)
        bits = {"a": int(a), "prev": int(prev)}
        # HLS and RTL units share the graph order (2-bit weights stay HLS)
        mvus = [n for n in rec["nodes"] if n["op"].startswith("MVAU")]
        assert len(mvus) == len(TOPOLOGY[g]), key
        for n, (mw, mh, src) in zip(mvus, TOPOLOGY[g]):
            pe, simd = n["fold"]["PE"], n["fold"]["SIMD"]
            assert n["cycles"] == (mw // simd) * (mh // pe) * OFM_PIXELS[g], (key, n["name"])
            if n["op"] != "MVAU_rtl":
                continue
            rb = replay_bram18(mw, mh, pe, simd, bits[src])
            for r in [n["auto"]] + n.get("variants", []):
                r["BRAM_18K"] = r["BRAM_18K"] - r.get("BRAM_replay", 0.0) + rb
                r["BRAM_replay"] = float(rb)
            n_set += 1
            n_bram += rb > 0
    d["replay_bram"] = {"min_bits": BRAM_MIN_BITS}
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {n_set} MVAU_rtl nodes, {n_bram} with a BRAM replay buffer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "tables"])
    p.add_argument("--audit")
    p.add_argument("--tables")
    p.add_argument("--out")
    args = p.parse_args()
    {"check": cmd_check, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
