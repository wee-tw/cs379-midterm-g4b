"""Re-run the ResNet streamlining transforms one at a time and report the first
one that leaves a node input with no producer (a dangling tensor)."""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, RemoveUnusedTensors, SortGraph)
from qonnx.transformation.infer_datatypes import InferDataTypes

sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
import finn_estimate_resnet8 as fe  # noqa: E402
from finn.transformation.streamline.reorder import MoveLinearPastEltwiseAdd, MoveLinearPastFork


def dangling(model):
    g = model.graph
    prod = {o for n in g.node for o in n.output}
    known = prod | {i.name for i in g.initializer} | {i.name for i in g.input}
    return [(n.op_type, n.name, i) for n in g.node for i in n.input if i and i not in known]


def consumers_of_fork(model):
    out = []
    for n in model.graph.node:
        for o in n.output:
            c = model.find_consumers(o)
            if c is not None and len(c) > 1:
                out.append((n.op_type, n.name, [x.op_type + ':' + x.name for x in c]))
    return out


model = ModelWrapper(sys.argv[1])
import inspect
src = inspect.getsource(fe.step_rn_streamline)
linear = [
    t for t in [
        fe.AbsorbScalarMulAddIntoTopK(), fe.ConvertSubToAdd(), fe.ConvertDivToMul(),
        fe.RemoveIdentityOps(), fe.CollapseRepeatedMul(), fe.BatchNormToAffine(),
        fe.ConvertSignToThres(), fe.MoveAddPastMul(), fe.MoveScalarAddPastMatMul(),
        fe.MoveAddPastConv(), fe.MoveScalarMulPastMatMul(), fe.MoveScalarMulPastConv(),
        fe.MoveScalarLinearPastInvariants(), fe.MoveAddPastMul(), fe.CollapseRepeatedAdd(),
        fe.CollapseRepeatedMul(), fe.AbsorbAddIntoMultiThreshold(), fe.FactorOutMulSignMagnitude(),
        fe.MoveMaxPoolPastMultiThreshold(), fe.AbsorbMulIntoMultiThreshold(),
        fe.Absorb1BitMulIntoMatMul(), fe.Absorb1BitMulIntoConv(), fe.RoundAndClipThresholds()]]
print("forks before streamlining:")
for f in consumers_of_fork(model):
    print("  ", f)
for it in range(4):
    for t in linear + [MoveLinearPastEltwiseAdd(), MoveLinearPastFork()]:
        before = [(n.op_type, n.name) for n in model.graph.node]
        model = model.transform(t)
        model = model.transform(GiveUniqueNodeNames())
        d = dangling(model)
        if d:
            print(f"iteration {it}: dangling input after {type(t).__name__}: {d}")
            print("nodes before:", before[:14])
            print("nodes after :", [(n.op_type, n.name) for n in model.graph.node][:14])
            sys.exit(0)
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
print("no dangling tensors")
