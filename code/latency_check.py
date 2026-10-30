"""Single-frame latency of the ResNet-8 designs (FINN critical path) and a check of the
rule used to compute it from the per-group cost tables.

FINN's dataflow_performance analysis gives critical_path_cycles = longest path of node
exp_cycles. The cost tables keep every node's cycles, so the same number can be built for
all design points: total cycles minus the shortcut branches (DownSampler + 1x1 MVAU in
stacks 2 and 3), provided the main branch is the longer one. This script checks that rule
against FINN on real folded models, then writes the latency of every row of points.npz.

  python latency_check.py check --tables cost/tables.json
  python latency_check.py points --tables cost/tables.json --points cost/points.npz \
      --out cost/latency.npz
"""

import argparse
import copy
import itertools
import os
import tempfile

import numpy as np

from assemble_costs import assemble, load_tables
from cost_tables import BITS, DSP_AWARE, MODES, dsp_aware_refold, hw_model


def crit_from_nodes(nodes):
    """Critical path (cycles) from a node list: shortcut branches are the DownSampler and
    the MVAU right after it in each downsampling stack, plus (standalone thresholds, 'dsp'
    mode) one of the two equal-cost Thresholding nodes that follow the two MVAUs."""
    shortcut = set()
    for i, n in enumerate(nodes):
        if n["op"].startswith("DownSampler"):
            shortcut.add(i)
            j = next(k for k in range(i + 1, len(nodes)) if nodes[k]["op"].startswith("MVAU"))
            shortcut.add(j)
            end = next(k for k in range(j + 1, len(nodes)) if nodes[k]["op"].startswith("FMPadding"))
            thr = [k for k in range(j + 1, end) if nodes[k]["op"].startswith("Thresholding")]
            if len(thr) >= 2:
                shortcut.add(thr[0])
    main = sum(n["cycles"] for i, n in enumerate(nodes) if i not in shortcut)
    return main, sum(nodes[i]["cycles"] for i in shortcut)


def finn_crit(m0, T, wwidth, dsp_aware=False):
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.infer_datatypes import InferDataTypes
    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth)
    from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
        MinimizeWeightBitWidth)
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    m = ModelWrapper(copy.deepcopy(m0.model))
    m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
    if dsp_aware:
        m = dsp_aware_refold(m)
    m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth())
    m = m.transform(InferDataTypes())
    return m.analysis(dataflow_performance)


def cmd_check(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    table, _ = load_tables(args.tables)
    tmp = tempfile.mkdtemp()
    for bits in ((4, 4, 4, 4, 4, 4, 8, 8), (4, 8, 4, 8, 4, 8, 4, 8), (8, 8, 2, 4, 2, 4, 8, 8)):
        for mode, (standalone, wwidth) in MODES.items():
            m0 = hw_model(bits, standalone, tmp)
            for T in (100_000, 25_000, 16_667, 10_000):
                perf = finn_crit(m0, T, wwidth, mode in DSP_AWARE)
                nodes = assemble(table, bits, mode, T)
                main, sc = crit_from_nodes(nodes)
                print(f"{''.join(map(str, bits))} {mode:8s} T={T:7d} FINN crit "
                      f"{perf['critical_path_cycles']:8d} max {perf['max_cycles']:7d} | rule "
                      f"{main:8d} (shortcut {sc:7d}) {'OK' if main == perf['critical_path_cycles'] else 'DIFF'}",
                      flush=True)


def cmd_points(args):
    table, _ = load_tables(args.tables)
    rows = np.load(args.points)["rows"]
    cfgs = list(itertools.product(BITS, repeat=8))
    modes = list(MODES)
    cache, lat = {}, np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        k = (int(r[0]), int(r[1]), int(r[2]))
        if k not in cache:
            cache[k] = crit_from_nodes(assemble(table, cfgs[k[0]], modes[k[1]], k[2]))[0]
        lat[i] = cache[k]
    np.savez(args.out, latency_cycles=lat)
    ratio = lat / rows[:, 3]
    print(f"{len(rows)} rows; critical path / bottleneck cycles: min {ratio.min():.2f} "
          f"median {np.median(ratio):.2f} max {ratio.max():.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "points"])
    p.add_argument("--tables", default="cost/v0/tables.json")
    p.add_argument("--points", default="cost/v0/points.npz")
    p.add_argument("--out", default="cost/v0/latency.npz")
    args = p.parse_args()
    {"check": cmd_check, "points": cmd_points}[args.cmd](args)


if __name__ == "__main__":
    main()
