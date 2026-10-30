"""Run the Vivado-free part of the FINN v0.10.1 flow on a ResNet-8 QONNX file.

Goes QONNX -> FINN-ONNX -> streamlined -> HW layers -> specialized -> folded,
then writes FINN's analytical estimate reports. No Vivado/Vitis is needed, so
this runs on a plain Python environment with FINN's src/ on PYTHONPATH.

  python finn_estimate_resnet8.py rn8.onnx --out build_rn8 --target-fps 2000

The tidy / streamline / convert-to-HW steps are copied from finn-examples
build/resnet50/custom_steps.py (residual adds need MoveLinearPastEltwiseAdd and
MoveLinearPastFork, which the default FINN streamlining step does not run).
"""

import argparse
import os

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul, ConvertSubToAdd, GiveReadableTensorNames, GiveUniqueNodeNames,
    GiveUniqueParameterTensors, RemoveStaticGraphInputs, RemoveUnusedTensors, SortGraph)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv, Absorb1BitMulIntoMatMul, AbsorbAddIntoMultiThreshold,
    AbsorbConsecutiveTransposes, AbsorbMulIntoMultiThreshold, AbsorbScalarMulAddIntoTopK,
    AbsorbTransposeIntoMultiThreshold, FactorOutMulSignMagnitude)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd, CollapseRepeatedMul)
from finn.transformation.streamline.reorder import (
    MoveAddPastConv, MoveAddPastMul, MoveLinearPastEltwiseAdd, MoveLinearPastFork,
    MoveMaxPoolPastMultiThreshold, MoveScalarAddPastMatMul, MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv, MoveScalarMulPastMatMul)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres


def apply(model, transforms):
    for t in transforms:
        model = model.transform(t)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_rn_tidy(model, cfg):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model.transform(InferDataTypes())


def step_rn_streamline(model, cfg):
    linear = [
        AbsorbScalarMulAddIntoTopK(), ConvertSubToAdd(), ConvertDivToMul(),
        RemoveIdentityOps(), CollapseRepeatedMul(), BatchNormToAffine(),
        ConvertSignToThres(), MoveAddPastMul(), MoveScalarAddPastMatMul(),
        MoveAddPastConv(), MoveScalarMulPastMatMul(), MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(), MoveAddPastMul(), CollapseRepeatedAdd(),
        CollapseRepeatedMul(), AbsorbAddIntoMultiThreshold(), FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(), AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(), Absorb1BitMulIntoConv(), RoundAndClipThresholds()]
    for _ in range(4):
        model = apply(model, linear)
        # Fork first: in v0.10.1, MoveLinearPastEltwiseAdd does not check whether the
        # Mul feeding the add also feeds another node. At ResNet-8's identity shortcut
        # (stem -> stack 1) it then deletes that Mul and leaves stack1.conv1 with a
        # dangling input. The ResNet-50 order (add first, then fork) hits exactly this.
        model = apply(model, [MoveLinearPastFork(), MoveLinearPastEltwiseAdd()])
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())
    return model.transform(DoubleToSingleFloat())


def step_rn_convert_to_hw(model, cfg):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    # Standalone thresholds keep MultiThresholds out of the MVAUs. Only then can
    # SpecializeLayers pick the DSP-based RTL MVU (it rejects embedded thresholds).
    # Lower and absorb the layout transposes first, as FINN's own step_streamline
    # does, and round the thresholds: the stem's are still float here, because the
    # input only becomes UINT8 in this step, and InferThresholdingLayer asserts ints.
    first = ([LowerConvsToMatMul, AbsorbTransposeIntoMultiThreshold, AbsorbConsecutiveTransposes,
              RoundAndClipThresholds, to_hw.InferThresholdingLayer]
             if cfg.standalone_thresholds else [])
    for t in first + [to_hw.InferAddStreamsLayer, LowerConvsToMatMul, to_hw.InferChannelwiseLinearLayer,
              to_hw.InferPool, AbsorbTransposeIntoMultiThreshold, RoundAndClipThresholds,
              to_hw.InferQuantizedMatrixVectorActivation, to_hw.InferThresholdingLayer,
              AbsorbConsecutiveTransposes, to_hw.InferConvInpGen,
              to_hw.InferDuplicateStreamsLayer, to_hw.InferLabelSelectLayer]:
        model = model.transform(t())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    non_hw = [n.op_type for n in model.graph.node if n.domain != "finn.custom_op.fpgadataflow"]
    print("nodes left outside fpgadataflow:", non_hw)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--out", default="build_rn8")
    p.add_argument("--target-fps", type=int, default=1000)
    p.add_argument("--clk-ns", type=float, default=10.0)
    p.add_argument("--standalone-thresholds", action="store_true")
    p.add_argument("--mvau-wwidth-max", type=int, default=36,
                   help="SetFolding cap on SIMD*W per MVAU (FINN default 36)")
    args = p.parse_args()
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=args.out,
        synth_clk_period_ns=args.clk_ns,
        board="Pynq-Z2",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        target_fps=args.target_fps,
        standalone_thresholds=args.standalone_thresholds,
        mvau_wwidth_max=args.mvau_wwidth_max,
        steps=["step_qonnx_to_finn", step_rn_tidy, step_rn_streamline, step_rn_convert_to_hw,
               "step_create_dataflow_partition", "step_specialize_layers",
               "step_target_fps_parallelization", "step_apply_folding_config",
               "step_minimize_bit_width", "step_generate_estimate_reports"],
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        save_intermediate_models=True,
        enable_build_pdb_debug=False,  # otherwise a failure waits forever in pdb
    )
    os.environ.setdefault("FINN_BUILD_DIR", os.path.abspath(args.out + "_tmp"))
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
    build.build_dataflow_cfg(args.model, cfg)


if __name__ == "__main__":
    main()
