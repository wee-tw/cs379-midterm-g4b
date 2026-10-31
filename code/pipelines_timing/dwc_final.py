"""Width converters (DWC) of the final designs, which the per-node costs leave out (notes 12.36).

generate: folds each design exactly as verify_designs.py generate does (SetFolding at T, the
          DSP-aware refold, the timing-safe refold except for lut_finn, minimized widths), runs
          FINN's InsertDWC and SpecializeLayers, and writes FINN's code of every inserted DWC as
          a case for verify_designs.py run (integer width ratios: RTL dwc.sv; others: the HLS
          LCM converter). The predicted LUT of a case is FINN's own DWC estimate.
report:   DWC LUT per design (FINN estimate and synthesis), and the energy per frame of A' and
          of the baselines of the main table with the synthesized DWC LUT added to the design.

  python dwc_final.py generate --designs vdw_designs.json --out D:/vdw --ts-rtl-simd 12 --ts-hls-simd 18
  python verify_designs.py run --out D:/vdw --workdir D:/vww --timing ...
  python dwc_final.py report --designs vdw_designs.json --out D:/vdw --table main_table_tsr_v1218_m0.json \
      --points 125=points_tsr_v1218_f125.npz 111.11=... 100=...
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def build(d, args):
    """The folded HW graph of design d with its DWCs inserted and specialized."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    import cost_tables as ct
    from verify_designs import PART
    bits, mode, T = d[0], d[1], d[2]
    standalone, wwidth = ct.MODES[mode]
    m0 = ct.hw_model(list(map(int, bits)), standalone, tempfile.mkdtemp())
    m = ModelWrapper(copy.deepcopy(m0.model))
    m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
    if mode in ct.DSP_AWARE:
        m = ct.dsp_aware_refold(m, mode in ct.DSP_CORRECTED)
    if args.ts_rtl_simd and mode != "lut_finn":   # as verify_designs.py generate --ts-refold
        os.environ["COST_TS_HLS_SIMD"] = str(args.ts_hls_simd or 0)
        m = ct.timing_safe_refold(m, args.ts_rtl_simd, mode in ct.DSP_CORRECTED)
    if args.narrow:   # MW:MH:PE:SIMD, e.g. the MW 576 layer at PE 64, SIMD 3 (notes 12.44)
        from qonnx.custom_op.registry import getCustomOp
        mw, mh, pe, simd = map(int, args.narrow.split(":"))
        hit = 0
        for n in m.graph.node:
            inst = getCustomOp(n) if n.op_type == "MVAU_rtl" else None
            if inst and inst.get_nodeattr("MW") == mw and inst.get_nodeattr("MH") == mh:
                lanes = inst.get_nodeattr("PE") * inst.get_nodeattr("SIMD")
                assert lanes == pe * simd, f"{n.name}: {lanes} lanes, not {pe * simd}"
                inst.set_nodeattr("PE", pe)
                inst.set_nodeattr("SIMD", simd)
                hit += 1
        assert hit == 1, f"--narrow matched {hit} MVAU_rtl nodes"
    m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
    m = m.transform(InferDataTypes())
    # InsertDWC leaves the new nodes unnamed; FINN's build names them the same way afterwards
    return m.transform(InsertDWC()).transform(SpecializeLayers(PART)).transform(GiveUniqueNodeNames())


def cmd_generate(args):
    from qonnx.custom_op.registry import getCustomOp
    from verify_designs import PART, tag
    designs = [tuple(d) for d in json.load(open(args.designs))]
    cases = []
    for di, d in enumerate(designs):
        m = build(d, args)
        f = d[3]
        dwcs = [n for n in m.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
        for i, n in enumerate(dwcs):
            assert n.name, f"unnamed DWC in {tag(d)}"
            inst = getCustomOp(n)
            name = f"d{di}_w{i:02d}_{n.name}"
            cdir = os.path.join(args.out, name)
            shutil.rmtree(cdir, ignore_errors=True)
            os.makedirs(cdir)
            inst.set_nodeattr("code_gen_dir_ipgen", os.path.abspath(cdir))
            kind = "rtl" if n.op_type.endswith("_rtl") else "hls"
            if kind == "rtl":
                inst.generate_hdl(m, PART, 1e3 / f)
            else:
                inst.code_generation_ipgen(m, PART, 1e3 / f)
            iw, ow = inst.get_nodeattr("inWidth"), inst.get_nodeattr("outWidth")
            cases.append({"case": name, "design": tag(d), "role": d[4], "op": n.op_type, "kind": kind,
                          "f_MHz": f, "predicted": {"LUT": float(inst.lut_estimation()),
                                                    "BRAM_18K": float(inst.bram_estimation()), "DSP": 0.0},
                          "attrs": {"inWidth": str(iw), "outWidth": str(ow)}})
        print(tag(d), d[4], len(dwcs), "DWCs:", " ".join(f"{c['attrs']['inWidth']}->{c['attrs']['outWidth']}"
                                                         f"{'' if c['kind'] == 'rtl' else '(hls)'}"
                                                         for c in cases if c["design"] == tag(d)), flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(cases, open(os.path.join(args.out, "cases.json"), "w"), indent=1)
    print(len(cases), "cases")


def cmd_report(args):
    import numpy as np
    import energy_search as es
    from verify_designs import tag
    designs = [tuple(d) for d in json.load(open(args.designs))]
    cases = json.load(open(os.path.join(args.out, "cases.json")))
    res = json.load(open(os.path.join(args.out, "results.json")))
    points = {float(k): np.load(v)["rows"] for k, v in (s.split("=", 1) for s in args.points)}
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    per = {}
    print("design                      role                    DWCs  hls  FINN LUT  synth LUT  synth FF  BRAM18 | "
          "design LUT  +DWC")
    for d in designs:
        cs = [c for c in cases if c["design"] == tag(d)]
        bad = [c["case"] for c in cs if res.get(c["case"], {}).get("returncode") != 0]
        finn = sum(c["predicted"]["LUT"] for c in cs)
        lut = sum(res[c["case"]]["LUT"] for c in cs if c["case"] not in bad)
        ff = sum(res[c["case"]].get("FF") or 0 for c in cs if c["case"] not in bad)
        bram = sum(res[c["case"]].get("BRAM18") or 0 for c in cs if c["case"] not in bad)
        rows = points[d[3]]
        mask = ((rows[:, 0].astype(int) == idx[d[0]]) & (rows[:, 1] == es.MODES.index(d[1]))
                & (rows[:, 2] == d[2]))
        row = rows[mask][:1].copy()
        per[d[0]] = (d, row, lut, bram)
        per_ns = [(res[c["case"]].get("period_ns") or 0, c["case"]) for c in cs if c["case"] not in bad] or [(0, "")]
        print(f"{tag(d):27s} {d[4]:23s} {len(cs):4d} {sum(c['kind'] == 'hls' for c in cs):4d} {finn:9.0f} "
              f"{lut:10.0f} {ff:9.0f} {bram:7.0f} | {int(row[0, 4]):10d} {100 * lut / row[0, 4]:+5.1f}%"
              f" | slowest DWC {max(per_ns)[0]:.2f} ns (clock {1e3 / d[3]:.2f} ns)"
              + (f"  NOT SYNTHESIZED: {bad}" if bad else ""))
    # FINN's DWC LUT estimate against synthesis, by direction (one row per distinct width pair)
    seen = {}
    for c in cases:
        if res.get(c["case"], {}).get("returncode") == 0:
            iw, ow = int(c["attrs"]["inWidth"]), int(c["attrs"]["outWidth"])
            seen[(c["op"][-3:], iw, ow)] = (c["predicted"]["LUT"], res[c["case"]]["LUT"])
    for name, sel in (("narrow -> wide", lambda k: k[1] < k[2]), ("wide -> narrow", lambda k: k[1] > k[2])):
        ks = [k for k in seen if sel(k)]
        for impl in ("rtl", "hls"):
            kk = [k for k in ks if k[0] == impl]
            if kk:
                f_, s_ = sum(seen[k][0] for k in kk), sum(seen[k][1] for k in kk)
                print(f"DWC {name:15s} {impl}: {len(kk):3d} width pairs, FINN LUT {f_:7.0f}, synthesis {s_:7.0f}")
    narrowed = {}   # A' bits -> DWC LUT with the layer narrowed (--narrow-out), converters matched by position
    for nout in args.narrow_out:
        ncases = json.load(open(os.path.join(nout, "cases.json")))
        nres = json.load(open(os.path.join(nout, "results.json")))
        for bits in {c["design"].split("_")[0] for c in ncases}:
            plain = {c["case"].split("_")[1]: c for c in cases if c["design"].split("_")[0] == bits}
            nar = {c["case"].split("_")[1]: c for c in ncases if c["design"].split("_")[0] == bits}
            assert sorted(plain) == sorted(nar), bits
            changed = [w for w in plain if plain[w]["attrs"] != nar[w]["attrs"]]
            delta = sum(nres[nar[w]["case"]]["LUT"] - res[plain[w]["case"]]["LUT"] for w in changed)
            narrowed[bits] = per[bits][2] + delta
            print(f"narrowed {bits}: DWC LUT {per[bits][2]:.0f} -> {narrowed[bits]:.0f}; "
                  + ", ".join(f"{plain[w]['attrs']['inWidth']}->{plain[w]['attrs']['outWidth']} "
                              f"({res[plain[w]['case']]['LUT']:.0f}) => {nar[w]['attrs']['inWidth']}->"
                              f"{nar[w]['attrs']['outWidth']} ({nres[nar[w]['case']]['LUT']:.0f})" for w in changed)
                  + f"; MVU LUT {args.narrow_lut:+d}")
    table = json.load(open(args.table))
    print("\nenergy per frame (mJ) with the synthesized DWC LUT and BRAM added; baseline above A' before -> after"
          + ("; [narrowed A': its MVU and DWC LUT change added]" if narrowed else ""))
    for p in ("nominal", "z2_anchor"):
        for m in sorted(table, key=float):
            r = table[m][p]

            def energy(bits, with_dwc, narrow=False):
                d, row, lut, bram = per[bits]
                x = row.copy()
                if with_dwc:
                    x[0, 4] += narrowed[bits] + args.narrow_lut if narrow else lut
                    x[0, 5] += bram
                return es.energies(x, es.POWER[p], d[3])[1][0] * 1e3
            a = r["A_verified"]["bits"]
            ea0, ea1 = energy(a, False), energy(a, True)
            ean = energy(a, True, True) if a in narrowed else None
            out = [f"A' {a} {ea0:.4f} -> {ea1:.4f} ({100 * (ea1 / ea0 - 1):+.1f}%)"
                   + (f" [narrowed {ean:.4f}, {100 * (ean / ea0 - 1):+.1f}%]" if ean else "")]
            for b in ("B1_measured", "B2_measured"):
                bb = r[b]["bits"]
                if bb not in per:
                    out.append(f"{b[:2]} {bb} not in designs")
                    continue
                e0, e1 = energy(bb, False), energy(bb, True)
                out.append(f"{b[:2]} {bb} {100 * (e0 / ea0 - 1):+.1f} -> {100 * (e1 / ea1 - 1):+.1f}% "
                           + (f"[{100 * (e1 / ean - 1):+.1f}%] " if ean else "")
                           + f"(table {100 * r[b]['rel_to_A']:+.1f}%)")
            print(f"{p:9s} margin {float(m):g}: " + " | ".join(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["generate", "report"])
    p.add_argument("--designs", required=True, help="[bits, mode, T, f_MHz, role, ...] per design")
    p.add_argument("--out", required=True)
    p.add_argument("--ts-rtl-simd", type=int, default=12)
    p.add_argument("--ts-hls-simd", type=int, default=18)
    p.add_argument("--only", type=int, default=None, help="generate: only design number N (smoke test)")
    p.add_argument("--narrow", default=None, help="generate: refold the one MVAU_rtl with this MW:MH to PE:SIMD")
    p.add_argument("--table", help="report: timing_safe.py table json")
    p.add_argument("--points", nargs="+", help="report: <f_MHz>=<points npz> per clock class")
    p.add_argument("--narrow-out", nargs="*", default=[],
                   help="report: dwc_final.py folder(s) of the A' designs generated with --narrow (only the "
                        "converters that changed need to be synthesized)")
    p.add_argument("--narrow-lut", type=int, default=436, help="report: routed LUT change of the narrowed MVU")
    args = p.parse_args()
    if args.cmd == "generate":
        if args.only is not None:
            designs = json.load(open(args.designs))
            tmp = os.path.join(tempfile.mkdtemp(), "one.json")
            json.dump([designs[args.only]], open(tmp, "w"))
            args.designs = tmp
        cmd_generate(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
