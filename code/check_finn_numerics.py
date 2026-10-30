"""Check that FINN's transformed ResNet-8 still computes what Brevitas computes.

Compares, on the same CIFAR-10 test images:
  1. Brevitas (PyTorch) logits, input = pixels / 255
  2. the exported QONNX graph, executed by qonnx (same float input)
  3. FINN after tidy + streamlining (input = raw UINT8 pixels, ToTensor merged in)
  4. FINN after conversion to HW layers (Python execution of each HW node)
Stages 3 and 4 end in TopK (top-1 index), so they are compared by predicted class.

  python check_finn_numerics.py --bits 4 4 4 4 4 4 8 8 --ckpt w4a4.pt --npz cifar10.npz
      [--recipe lowbit] [--n 20]
"""

import argparse
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup_model

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.pytorch import ToTensor

import finn_estimate_resnet8 as fe
from resnet8_brevitas import ResNet8
from verify_resnet8 import parse_bits


def run(model, x):
    name = model.graph.input[0].name
    return execute_onnx(model, {name: x})[model.graph.output[0].name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, nargs=8, default=[4] * 6 + [8, 8])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--npz", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--recipe", default="default",
                   help="quantizer recipe the checkpoint was trained with (train_cifar.py --recipe)")
    args = p.parse_args()

    net = ResNet8(parse_bits(args.bits), recipe=args.recipe)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()
    d = np.load(args.npz)
    x_u8 = d["x_test"][:args.n].transpose(0, 3, 1, 2).astype(np.float32)  # NCHW, 0..255
    y = d["y_test"][:args.n]

    tmp = tempfile.mkdtemp()
    qonnx_path = os.path.join(tmp, "rn8.onnx")
    export_qonnx(net, torch.randn(1, 3, 32, 32), export_path=qonnx_path)
    pre_path = os.path.join(tmp, "pre.onnx")
    export_qonnx(ToTensor(), torch.randn(1, 3, 32, 32), export_path=pre_path)

    qonnx_model = cleanup_model(ModelWrapper(qonnx_path))
    finn_model = qonnx_model.transform(ConvertQONNXtoFINN())
    pre = cleanup_model(ModelWrapper(pre_path)).transform(ConvertQONNXtoFINN())
    finn_model = finn_model.transform(MergeONNXModels(pre))
    finn_model.set_tensor_datatype(finn_model.graph.input[0].name, DataType["UINT8"])
    cfg = SimpleNamespace(standalone_thresholds=False)
    tidy = fe.step_rn_tidy(finn_model, cfg)
    streamlined = fe.step_rn_streamline(tidy, cfg)
    hw = fe.step_rn_convert_to_hw(ModelWrapper(streamlined.model), cfg)

    agree = {"qonnx": 0, "streamlined": 0, "hw": 0}
    max_diff, correct, classes = 0.0, 0, set()
    for i in range(args.n):
        xf = x_u8[i:i + 1] / 255.0
        with torch.no_grad():
            ref = net(torch.from_numpy(xf)).numpy()
        cls = int(ref.argmax())
        classes.add(cls)
        correct += int(cls == y[i])
        q = run(qonnx_model, xf.astype(np.float32))
        max_diff = max(max_diff, float(np.abs(q - ref).max()))
        agree["qonnx"] += int(int(q.argmax()) == cls)
        agree["streamlined"] += int(int(run(streamlined, x_u8[i:i + 1]).flatten()[0]) == cls)
        agree["hw"] += int(int(run(hw, x_u8[i:i + 1].transpose(0, 2, 3, 1)
                               if hw.get_tensor_layout(hw.graph.input[0].name) == "NHWC"
                               else x_u8[i:i + 1]).flatten()[0]) == cls)
    print(f"images={args.n}  brevitas top-1 correct={correct}  distinct predicted classes={len(classes)}")
    print(f"max |QONNX - Brevitas| logit difference = {max_diff:.2e}")
    for k, v in agree.items():
        print(f"{k:12s} predicts the same class as Brevitas on {v}/{args.n} images")


if __name__ == "__main__":
    main()
