"""Per-group FINN v0.10.1 cost tables for ResNet-8, for every bit-width context.

FINN's SetFolding folds each layer independently to meet a target number of cycles
per frame, and its resource estimates are per node. So the cost of a group depends
only on its own (W, A) and the activation width of the previous group (the input of
its first layer). This script builds those per-group records from a small covering
set of configurations and saves them; assemble_costs.py then adds them up for any of
the 9^4 = 6,561 configurations.

For every HW node it records the expected cycles and FINN's resource estimate for
each implementation variant FINN lets us choose per node:
  MVAU_hls: resType lut/dsp x weight ram_style block/distributed x
            threshold ram_style block/distributed
  MVAU_rtl (DSP-based, needs standalone thresholds): weight ram_style block/distributed
  ConvolutionInputGenerator_rtl: line-buffer ram_style block/distributed
plus FINN's own "auto" choice.

Modes (how the network is converted and folded):
  lut       thresholds inside the MVAUs (MVAU_hls only), SetFolding cap SIMD*W <= 1024
  dsp       standalone thresholds, so >=4-bit MVAUs become DSP-based MVAU_rtl, cap 1024
  lut_finn  as lut, but with FINN's default SetFolding cap SIMD*W <= 36
  dsp4      as dsp, then DSP-aware refolding of every MVAU_rtl (see dsp_aware_refold)
  dspc      as dsp, but MVAU_rtl DSPs counted as the RTL instantiates them (rtl_mvu_dsps)
  dsp4c     as dsp4 with that count, refolded for it

  python cost_tables.py --out tables.json --workers 5 [--modes dsp4]
"""

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from types import SimpleNamespace

T_GRID = [1_000_000, 500_000, 200_000, 100_000, 50_000, 33_333, 25_000, 20_000,
          16_667, 14_286, 12_500, 10_000]
MODES = {"lut": (False, 1024), "dsp": (True, 1024), "lut_finn": (False, 36),
         "dsp4": (True, 1024), "dspc": (True, 1024), "dsp4c": (True, 1024)}
DSP_AWARE = {"dsp4", "dsp4c"}
DSP_CORRECTED = {"dspc", "dsp4c"}
BITS = (2, 4, 8)
VARIANTS = {
    "MVAU_hls": {"resType": ["lut", "dsp"], "ram_style": ["block", "distributed"],
                 "ram_style_thresholds": ["block", "distributed"]},
    "MVAU_rtl": {"ram_style": ["block", "distributed"]},
    "ConvolutionInputGenerator_rtl": {"ram_style": ["block", "distributed"]},
}


def covering_configs():
    """Every (W_g, A_g, A_{g-1}) context appears: all 81 A-patterns x uniform W."""
    out = []
    for a in itertools.product(BITS, repeat=4):
        for w in BITS:
            out.append([w, a[0], w, a[1], w, a[2], w, a[3]])
    return out


def contexts(bits):
    """Per-group key (W, A, A_prev); the network input is UINT8."""
    w = bits[0::2]
    a = bits[1::2]
    prev = [8, a[0], a[1], a[2]]
    return [(w[g], a[g], prev[g]) for g in range(4)]


def compact(r):
    return {k: float(r.get(k, 0)) for k in ("LUT", "BRAM_18K", "URAM", "DSP")}


def hw_model(bits, standalone, tmp, ckpt=None):
    import torch
    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.util.cleanup import cleanup_model
    from finn.transformation.fpgadataflow.create_dataflow_partition import (
        CreateDataflowPartition)
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    from finn.util.basic import pynq_part_map
    import finn_estimate_resnet8 as fe
    from resnet8_brevitas import ResNet8
    from verify_resnet8 import parse_bits

    torch.manual_seed(0)
    if ckpt:  # trained weights (the cost tables use the seeded random init); the
        # checkpoints come from train_cifar.py --recipe lowbit (learnable weight scales)
        net = ResNet8(parse_bits(bits), recipe="lowbit").eval()
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        net = ResNet8(parse_bits(bits)).eval()
    path = os.path.join(tmp, "m.onnx")
    export_qonnx(net, torch.randn(1, 3, 32, 32), export_path=path)
    m = cleanup_model(ModelWrapper(path)).transform(ConvertQONNXtoFINN())
    cfg = SimpleNamespace(standalone_thresholds=standalone)
    m = fe.step_rn_tidy(m, cfg)
    m = fe.step_rn_streamline(m, cfg)
    m = fe.step_rn_convert_to_hw(m, cfg)
    # like the builder: keep only the dataflow partition (drops the input Transpose,
    # which SetFolding's dataflow_performance analysis cannot handle)
    os.environ["FINN_BUILD_DIR"] = tmp
    parent = m.transform(CreateDataflowPartition(partition_model_dir=os.path.join(tmp, "dp")))
    sdp = parent.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    m = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))
    m = m.transform(SpecializeLayers(pynq_part_map["Pynq-Z2"]))
    return m.transform(GiveUniqueNodeNames())


def hw_nodes(m):
    return [n for n in m.graph.node if n.domain.startswith("finn.custom_op.fpgadataflow")]


def assign_groups(m):
    """Group 0..3 by position relative to the post-add Thresholding nodes."""
    nodes = hw_nodes(m)
    add_outs = {n.output[0] for n in nodes if n.op_type.startswith("AddStreams")}
    g, groups = 0, {}
    for n in nodes:  # graph order is topological (SortGraph in the convert step)
        groups[n.name] = g
        if n.op_type.startswith("Thresholding") and n.input[0] in add_outs:
            g += 1
    return groups


def rtl_pack(inst, corrected):
    """PEs per DSP48 of a v0.10.1 MVAU_rtl. finn-rtllib/mvu instantiates one DSP per SIMD
    lane for every ceil(PE / 4) PEs in mvu_4sx4u (weights and activations <= 4 bits) but
    every ceil(PE / 2) PEs in mvu_8sx8u_dsp48 (anything wider; PIPE_COUNT in both files).
    FINN's dsp_estimation uses PE / 4 for both, i.e. half the DSPs of the 8-bit core."""
    if not corrected:
        return 4
    a, w = inst.get_input_datatype(0).bitwidth(), inst.get_input_datatype(1).bitwidth()
    return 4 if (a <= 4 and w <= 4) else 2


def rtl_mvu_dsps(inst):
    pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
    return math.ceil(pe / rtl_pack(inst, True)) * simd


def rtl_mvu_luts(n):
    """(low, high) LUT estimates of a v0.10.1 MVAU_rtl. Its own lut_estimation returns 0,
    which also makes LUTRAM weights look free. Low adds only the LUTRAM of distributed
    weights; high is FINN's estimate for the same layer as a DSP-based HLS MVAU (resType
    dsp, same folding and weight memory: base, adder tree, accumulator, LUTRAM), which
    overstates the RTL core, whose adder chain sits in the DSP cascade."""
    from qonnx.custom_op.registry import getCustomOp
    h = copy.deepcopy(n)
    h.op_type, h.domain = "MVAU_hls", "finn.custom_op.fpgadataflow.hls"
    inst = getCustomOp(h)
    inst.set_nodeattr("resType", "dsp")
    out = {}
    for style in ("block", "distributed"):
        inst.set_nodeattr("ram_style", style)
        out[style] = float(inst.lut_estimation())
    lutram = out["distributed"] - out["block"]
    return {"block": (0.0, out["block"]), "distributed": (lutram, out["distributed"])}


def dsp_aware_refold(m, corrected=False):
    """FINN v0.10.1 estimates a DSP48E1 MVAU_rtl at ceil(PE/4)*SIMD DSPs (four PEs share
    a DSP column), but SetFolding raises SIMD first, up to SIMD*W <= cap. Narrow weights
    therefore get wide-SIMD, narrow-PE layers that leave columns half empty (4-bit stack-1
    convs: PE 2, SIMD 144, 144 DSPs; 8-bit: PE 4, SIMD 72, 72 DSPs). Keep PE*SIMD, and so
    the cycles, and move lanes from SIMD to PE where the divisors allow, minimising the
    DSP estimate and otherwise staying closest to SetFolding's PE."""
    from qonnx.custom_op.registry import getCustomOp
    for n in m.graph.node:
        if n.op_type != "MVAU_rtl":
            continue
        inst = getCustomOp(n)
        mw, mh = inst.get_nodeattr("MW"), inst.get_nodeattr("MH")
        pe0, simd0 = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
        lanes = pe0 * simd0
        per = rtl_pack(inst, corrected)
        opts = [(math.ceil(pe / per) * (lanes // pe), abs(pe - pe0), pe, lanes // pe)
                for pe in range(1, mh + 1)
                if mh % pe == 0 and lanes % pe == 0 and mw % (lanes // pe) == 0]
        _, _, pe, simd = min(opts)
        inst.set_nodeattr("PE", pe)
        inst.set_nodeattr("SIMD", simd)
    return m


def fold_and_estimate(m0, T, wwidth, dsp_aware=False, corrected=False):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    from finn.util.basic import pynq_part_map
    part = pynq_part_map["Pynq-Z2"]

    m = ModelWrapper(copy.deepcopy(m0.model))
    # same order as the FINN builder: fold, then minimize bit-widths, then estimate.
    # two_pass_relaxation=False keeps every layer's folding independent of the rest of
    # the network (the builder default re-targets all layers to the network bottleneck,
    # which couples the groups); sweeping T gives the same designs.
    m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
    if dsp_aware:
        m = dsp_aware_refold(m, corrected)
    m = m.transform(MinimizeWeightBitWidth())
    m = m.transform(MinimizeAccumulatorWidth())
    m = m.transform(InferDataTypes())
    groups = assign_groups(m)
    rows = []
    for n in hw_nodes(m):
        inst = getCustomOp(n)
        row = {"name": n.name, "op": n.op_type, "group": groups[n.name],
               "cycles": int(inst.get_exp_cycles()),
               "auto": compact(inst.node_res_estimation(part)),
               "fold": {k: inst.get_nodeattr(k) for k in ("PE", "SIMD")
                        if k in inst.get_nodeattr_types()}}
        spec = VARIANTS.get(n.op_type)
        if spec:
            keys = [k for k in spec if k in inst.get_nodeattr_types()]
            orig = {k: inst.get_nodeattr(k) for k in keys}
            row["variants"] = []
            for combo in itertools.product(*[spec[k] for k in keys]):
                for k, v in zip(keys, combo):
                    inst.set_nodeattr(k, v)
                row["variants"].append({"attrs": dict(zip(keys, combo)),
                                        **compact(inst.node_res_estimation(part))})
            for k, v in orig.items():
                inst.set_nodeattr(k, v)
        if corrected and n.op_type == "MVAU_rtl":
            dsp = float(rtl_mvu_dsps(inst))
            luts = rtl_mvu_luts(n)
            style = inst.get_nodeattr("ram_style")
            lo, hi = luts.get(style, luts["block"])
            row["auto"].update({"DSP": dsp, "LUT": lo, "LUT_hi": hi})
            for v in row.get("variants", []):
                lo, hi = luts[v["attrs"]["ram_style"]]
                v.update({"DSP": dsp, "LUT": lo, "LUT_hi": hi})
        rows.append(row)
    return rows


def run_config(bits):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tmp = tempfile.mkdtemp()
    out = []
    built = {}
    wanted = os.environ.get("COST_MODES")  # env var: Windows workers are spawned, not forked
    for mode, (standalone, wwidth) in MODES.items():
        if wanted and mode not in wanted.split(","):
            continue
        if standalone not in built:
            built[standalone] = hw_model(bits, standalone, tmp)
        for T in T_GRID:
            rows = fold_and_estimate(built[standalone], T, wwidth, mode in DSP_AWARE,
                                     mode in DSP_CORRECTED)
            for g, ctx in enumerate(contexts(bits)):
                grows = [r for r in rows if r["group"] == g]
                out.append({"mode": mode, "T": T, "group": g, "ctx": ctx, "bits": bits,
                            "cycles": max(r["cycles"] for r in grows), "nodes": grows})
    return out


def key(rec):
    w, a, p = rec["ctx"]
    return f'{rec["mode"]}|{rec["T"]}|{rec["group"]}|{w}|{a}|{p}'


def total(rec):
    return {k: sum(r["auto"][k] for r in rec["nodes"]) for k in ("LUT", "BRAM_18K", "DSP")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only the first N configs (test)")
    p.add_argument("--modes", default=None, help="comma-separated subset of MODES")
    args = p.parse_args()
    if args.modes:
        os.environ["COST_MODES"] = args.modes
    cfgs = covering_configs()[:args.limit]
    table, mismatches, t0 = {}, [], time.time()
    with mp.Pool(args.workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(run_config, cfgs)):
            for rec in recs:
                k = key(rec)
                if k not in table:
                    table[k] = rec
                elif total(table[k]) != total(rec) or table[k]["cycles"] != rec["cycles"]:
                    mismatches.append((k, rec["bits"], table[k]["bits"]))
            print(f"{i + 1}/{len(cfgs)} configs, {len(table)} group records, "
                  f"{len(mismatches)} mismatches, {time.time() - t0:.0f} s", flush=True)
    json.dump({"T_GRID": T_GRID, "modes": args.modes.split(",") if args.modes else list(MODES),
               "table": table,
               "mismatches": mismatches}, open(args.out, "w"))
    print("saved", args.out, "records", len(table), "mismatches", len(mismatches))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
