"""Latency-aware folding of the cheap residual-path nodes (FINN v0.10.1 estimates).

SetFolding folds every node just enough to meet the cycle target T, so the residual
plumbing (DuplicateStreams, AddStreams and the Thresholding after each add) also runs at
about T cycles per frame and makes up a third of the single-frame latency of the selected
ResNet-8 design. This script folds the design as the cost tables do, then raises PE of
those nodes to their channel count and reports FINN's critical path, max cycles and
resource estimates before and after.

  python latency_fold.py --bits 4 4 4 4 4 4 8 8 --T 16667
"""

import argparse
import copy
import os
import tempfile

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from cost_tables import hw_model

CHEAP = ("DuplicateStreams", "AddStreams", "Thresholding")


def fold(m0, T, wwidth=1024):
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    m = ModelWrapper(copy.deepcopy(m0.model))
    m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
    m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
    return m.transform(InferDataTypes())


def report(m, tag):
    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
    from finn.util.basic import pynq_part_map
    part = pynq_part_map["Pynq-Z2"]
    # dataflow_performance reads the cycles_estimate attribute: re-annotate after PE edits
    m = m.transform(AnnotateCycles())
    perf = m.analysis(dataflow_performance)
    tot = {"LUT": 0.0, "BRAM_18K": 0.0, "DSP": 0.0}
    cheap = {"LUT": 0.0, "BRAM_18K": 0.0, "DSP": 0.0}
    for n in m.graph.node:
        if not n.domain.startswith("finn.custom_op.fpgadataflow"):
            continue
        r = getCustomOp(n).node_res_estimation(part)
        for k in tot:
            tot[k] += r.get(k, 0)
            if n.op_type.startswith(CHEAP):
                cheap[k] += r.get(k, 0)
    print(f"{tag:10s} critical path {perf['critical_path_cycles']:8d}  max cycles "
          f"{perf['max_cycles']:7d}  LUT {tot['LUT']:8.0f} (residual nodes {cheap['LUT']:6.0f})  "
          f"BRAM18 {tot['BRAM_18K']:5.0f}  DSP {tot['DSP']:4.0f}", flush=True)
    return perf, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs=8, default=[4, 4, 4, 4, 4, 4, 8, 8])
    p.add_argument("--T", type=int, default=16667)
    args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    m0 = fold(hw_model(args.bits, False, tempfile.mkdtemp()), args.T)
    base, rb = report(m0, "SetFolding")
    for div in (4, 2, 1):            # PE = C/div on every residual-path node
        m = ModelWrapper(copy.deepcopy(m0.model))
        for n in m.graph.node:
            if n.op_type.startswith(CHEAP) and n.domain.startswith("finn.custom_op.fpgadataflow"):
                inst = getCustomOp(n)
                pe = max(1, inst.get_nodeattr("NumChannels") // div)
                if pe > inst.get_nodeattr("PE"):
                    inst.set_nodeattr("PE", pe)
        new, rn = report(m, f"PE = C/{div}")
        print(f"   critical path {100 * (new['critical_path_cycles'] / base['critical_path_cycles'] - 1):+.1f}%, "
              f"max cycles {new['max_cycles']} vs {base['max_cycles']}, "
              f"LUT {100 * (rn['LUT'] / rb['LUT'] - 1):+.1f}% ({rn['LUT'] - rb['LUT']:+.0f})")


if __name__ == "__main__":
    main()
