"""What the per-node costs leave out: width converters and the residual skip FIFOs
(implementation_notes.md 12.36).

DWC: builds the design like cost_tables.py (same conversion, SetFolding at T, DSP-aware refold
for dsp4c), runs FINN's InsertDWC and SpecializeLayers, and lists every inserted
StreamingDataWidthConverter with FINN's own estimate.
Skip FIFO (analytical): the shortcut branch of each stack has to hold what arrives while the
main branch fills its two 3x3 sliding windows (about K-1 rows plus K pixels each). With every
layer running at the frame rate, that is a fraction rows_filled / rows_per_frame of a frame of
the shortcut stream at its output.

  python fifo_dwc_estimate.py 44444448:dsp4c:16667 84848484:lut:16667
"""

import math
import os
import sys
import tempfile

# (stack, input H=W, C_in, C_out, stride): ResNet-8 stacks 1-3
STACKS = [(1, 32, 16, 16, 1), (2, 32, 16, 32, 2), (3, 16, 32, 64, 2)]
BRAM18_BITS = 18 * 1024


def skip_fifo(bits):
    """Rough skip-FIFO size per stack: (elements, bits of storage, BRAM18 if put in BRAM)."""
    a = [int(c) for c in bits[1::2]]          # A1..A4 of the groups
    out = []
    for k, (s, hw, cin, cout, stride) in enumerate(STACKS):
        h2 = hw // stride
        frac = 2 / (hw + 2) + 2 / (h2 + 2)     # two SWG fills, as a share of a frame
        elems = math.ceil(frac * h2 * h2 * cout)
        width = a[k]                           # the stack's own activation width
        out.append((s, elems, elems * width, math.ceil(elems * width / BRAM18_BITS)))
    return out


def dwcs(bits, mode, T):
    from qonnx.custom_op.registry import getCustomOp
    from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
    from finn.transformation.fpgadataflow.set_folding import SetFolding
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.util.basic import pynq_part_map
    import cost_tables as ct
    part = pynq_part_map["Pynq-Z2"]
    standalone, wwidth = ct.MODES[mode]
    m = ct.hw_model([int(c) for c in bits], standalone, tempfile.mkdtemp())
    m = m.transform(SetFolding(T, mvau_wwidth_max=wwidth, two_pass_relaxation=False))
    if mode in ct.DSP_AWARE:
        m = ct.dsp_aware_refold(m, mode in ct.DSP_CORRECTED)
    m = m.transform(InsertDWC()).transform(SpecializeLayers(part))
    rows = []
    for n in m.graph.node:
        if n.op_type.startswith("StreamingDataWidthConverter"):
            inst = getCustomOp(n)
            est = inst.node_res_estimation(part)
            rows.append((n.name, n.op_type, inst.get_nodeattr("inWidth"), inst.get_nodeattr("outWidth"),
                         est.get("LUT", 0), est.get("BRAM_18K", 0)))
    return rows


def main():
    for spec in sys.argv[1:]:
        bits, mode, T = spec.split(":")
        print(f"== {bits} {mode} T{T}")
        rows = dwcs(bits, mode, int(T))
        for r in rows:
            print(f"   {r[0]:40s} {r[1]:36s} {r[2]:5d} -> {r[3]:5d} bits | FINN LUT {r[4]:7.0f} BRAM18 {r[5]:3.0f}")
        print(f"   DWCs: {len(rows)}, FINN LUT {sum(r[4] for r in rows):.0f}, BRAM18 {sum(r[5] for r in rows):.0f}")
        for s, elems, nbits, b18 in skip_fifo(bits):
            print(f"   skip FIFO stack {s}: ~{elems} elements, {nbits / 1024:.1f} kbit, {b18} BRAM18")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
