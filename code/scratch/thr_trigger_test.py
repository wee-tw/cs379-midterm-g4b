"""Check FINN v0.10.1 Thresholding_rtl memory estimates under its depth_trigger attributes."""
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import cost_tables as ct  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from qonnx.transformation.infer_datatypes import InferDataTypes  # noqa: E402
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth  # noqa: E402
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth  # noqa: E402
from finn.transformation.fpgadataflow.set_folding import SetFolding  # noqa: E402
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal  # noqa: E402

PART = "xc7z020clg400-1"
m = ct.hw_model([8, 8, 8, 8, 8, 8, 8, 8], True, tempfile.mkdtemp())
m = m.transform(SetFolding(100000, mvau_wwidth_max=1024, two_pass_relaxation=False))
m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth()).transform(InferDataTypes())
for n in m.graph.node:
    if n.op_type != "Thresholding_rtl":
        continue
    inst = getCustomOp(n)
    if inst.get_nodeattr("NumChannels") != 64 or "UINT8" not in inst.get_nodeattr("outputDataType"):
        continue
    print(n.name, "PE", inst.get_nodeattr("PE"), "C", inst.get_nodeattr("NumChannels"),
          inst.get_nodeattr("inputDataType"), "->", inst.get_nodeattr("outputDataType"))
    print("  geometries (width, depth):", inst.get_pe_mem_geometries())
    for tb, tu in ((0, 0), (1024, 0), (1024, 1 << 30), (0, 1 << 30)):
        inst.set_nodeattr("depth_trigger_bram", tb)
        inst.set_nodeattr("depth_trigger_uram", tu)
        est = {k: v for k, v in inst.node_res_estimation(PART).items() if k in ("LUT", "BRAM_18K", "URAM")}
        print(f"  trigger bram {tb} uram {tu}: get_memory_estimate {inst.get_memory_estimate()} -> node_res_estimation {est}")
    # what the corrected rule (dev: BRAM key fixed; uram trigger 0 meaning off) gives at D = 1024
    bram = lut = 0
    for w, d in inst.get_pe_mem_geometries():
        prims = {k: v for k, v in mem_primitives_versal.items() if ("BRAM" in k if d >= 1024 else True)
                 and "URAM" not in k}
        name, (cnt, eff, waste) = get_memutil_alternatives((w, d), prims)[0]
        if name.startswith("BRAM"):
            bram += cnt
        else:
            lut += cnt
    print(f"  intended at D = 1024 (no URAM on xc7z020): BRAM18 {bram * inst.get_nodeattr('PE')}, LUTRAM {lut * inst.get_nodeattr('PE')}")
    break
