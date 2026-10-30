"""Does the MoveLinearPastEltwiseAdd fork problem also hit Brevitas' own FINN-ready
ResNet-18 (bnn_pynq quant_resnet18, 4w4a)? Streamline it with the finn-examples
ResNet-50 order (add first, then fork) and with the swapped order, and count node inputs
that no node produces (dangling tensors)."""
import sys
import tempfile
from types import SimpleNamespace

import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models import model_with_cfg
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.cleanup import cleanup_model

sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
import finn_estimate_resnet8 as fe  # noqa: E402
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN  # noqa: E402
from finn.transformation.streamline.reorder import (  # noqa: E402
    MoveLinearPastEltwiseAdd, MoveLinearPastFork)


def linear():  # same list as finn_estimate_resnet8.step_rn_streamline
    return [fe.AbsorbScalarMulAddIntoTopK(), fe.ConvertSubToAdd(), fe.ConvertDivToMul(),
            fe.RemoveIdentityOps(), fe.CollapseRepeatedMul(), fe.BatchNormToAffine(),
            fe.ConvertSignToThres(), fe.MoveAddPastMul(), fe.MoveScalarAddPastMatMul(),
            fe.MoveAddPastConv(), fe.MoveScalarMulPastMatMul(), fe.MoveScalarMulPastConv(),
            fe.MoveScalarLinearPastInvariants(), fe.MoveAddPastMul(), fe.CollapseRepeatedAdd(),
            fe.CollapseRepeatedMul(), fe.AbsorbAddIntoMultiThreshold(),
            fe.FactorOutMulSignMagnitude(), fe.MoveMaxPoolPastMultiThreshold(),
            fe.AbsorbMulIntoMultiThreshold(), fe.Absorb1BitMulIntoMatMul(),
            fe.Absorb1BitMulIntoConv(), fe.RoundAndClipThresholds()]


def dangling(m):
    g = m.graph
    known = ({o for n in g.node for o in n.output} | {i.name for i in g.initializer}
             | {i.name for i in g.input})
    return [(n.op_type, n.name, i) for n in g.node for i in n.input if i and i not in known]


net, _ = model_with_cfg("resnet18_4w4a", pretrained=False)
net.eval()
path = tempfile.mkdtemp() + "/r18.onnx"
export_qonnx(net, torch.randn(1, 3, 32, 32), export_path=path)
base = cleanup_model(ModelWrapper(path)).transform(ConvertQONNXtoFINN())
base = fe.step_rn_tidy(base, SimpleNamespace(standalone_thresholds=False))
for name, nonlinear in [("add then fork (finn-examples ResNet-50 order)",
                         lambda: [MoveLinearPastEltwiseAdd(), MoveLinearPastFork()]),
                        ("fork then add (our order)",
                         lambda: [MoveLinearPastFork(), MoveLinearPastEltwiseAdd()])]:
    m = ModelWrapper(base.model)
    for _ in range(4):
        m = fe.apply(m, linear())
        m = fe.apply(m, nonlinear())
    d = dangling(m)
    print(f"{name}: {len(d)} dangling inputs {d[:3]}")
