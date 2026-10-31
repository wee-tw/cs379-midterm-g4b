import sys, tempfile
sys.path.insert(0, sys.argv[1])
import cost_tables as ct
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.set_folding import SetFolding
m0 = ct.hw_model([8, 8, 8, 8, 8, 8, 8, 8], True, tempfile.mkdtemp())
for T in (100000, 12500):
    m = m0.transform(SetFolding(T, mvau_wwidth_max=1024, two_pass_relaxation=False))
    m = m.transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth()).transform(InferDataTypes())
    for n in m.graph.node:
        if n.op_type != "Thresholding_rtl":
            continue
        i = getCustomOp(n)
        thr = m.get_initializer(n.input[1])
        print(T, n.name, "PE", i.get_nodeattr("PE"), "C", i.get_nodeattr("NumChannels"), i.get_nodeattr("inputDataType"),
              "->", i.get_nodeattr("outputDataType"), "uniform", i.get_nodeattr("uniform_thres"), "wdt", i.get_weight_datatype(),
              "thr shape", None if thr is None else thr.shape, "LUT", i.lut_estimation(), "geom", i.get_pe_mem_geometries()[:3])
