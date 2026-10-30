"""Checks for the Brevitas ResNet-8 before any FPGA work.

  python verify_resnet8.py count                 # weights / MACs per layer
  python verify_resnet8.py export --bits 4 4 4 4 4 4 8 8 --out rn8.onnx [--ckpt w.pt]
  python verify_resnet8.py bench  --bits 4 4 4 4 4 4 8 8   # GPU training speed

--bits takes W1 A1 W2 A2 W3 A3 W4 A4 (G1..G4).
"""

import argparse
import time

import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet8_brevitas import ResNet8, count_weights_and_macs


def parse_bits(flat):
    return tuple((flat[i], flat[i + 1]) for i in range(0, 8, 2))


def cmd_count(args):
    rows = count_weights_and_macs(ResNet8(parse_bits(args.bits)))
    for name, w, macs in rows:
        print(f'{name:28s} weights={w:7d}  MACs={macs:10d}')
    print(f'TOTAL weights={sum(r[1] for r in rows)}  MACs={sum(r[2] for r in rows)}')


def quant_params(model, tensor_name, producers, inits):
    """Follow a tensor back to the QONNX Quant node that produced it."""
    node = producers.get(tensor_name)
    while node is not None and node.op_type != 'Quant':
        node = producers.get(node.input[0])
    if node is None:
        return None
    scale, zp, bw = (inits.get(n) for n in node.input[1:4])
    signed = next(a.i for a in node.attribute if a.name == 'signed')
    return node.name, scale, zp, bw, signed


def cmd_export(args):
    from brevitas.export import export_qonnx
    model = ResNet8(parse_bits(args.bits))
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    export_qonnx(model, torch.randn(1, 3, 32, 32), export_path=args.out)
    g = onnx.load(args.out).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producers = {o: n for n in g.node for o in n.output}
    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print('op counts:', ops)
    for n in g.node:
        if n.op_type != 'Add' or any(i in inits for i in n.input):
            continue
        a, b = (quant_params(model, i, producers, inits) for i in n.input)
        same_node = a is not None and b is not None and a[0] == b[0]
        same_vals = (a is not None and b is not None and
                     all((x == y).all() for x, y in zip(a[1:4], b[1:4])) and a[4] == b[4])
        print(f'{n.name}: inputs from {a and a[0]} / {b and b[0]}; '
              f'same Quant node={same_node}, same scale/zp/bits/sign={same_vals}, '
              f'bits={a and float(a[3])}, signed={a and a[4]}')


class FloatResNet8(nn.Module):
    """Plain PyTorch ResNet-8 (MLPerf Tiny topology) for an FP speed reference."""

    def __init__(self):
        super().__init__()
        c = lambda i, o, k, s: nn.Conv2d(i, o, k, s, k // 2, bias=False)
        self.stem, self.bn0 = c(3, 16, 3, 1), nn.BatchNorm2d(16)
        cfg = [(16, 16, 1), (16, 32, 2), (32, 64, 2)]
        self.convs = nn.ModuleList()
        for i, o, s in cfg:
            self.convs.append(nn.ModuleList([
                c(i, o, 3, s), nn.BatchNorm2d(o), c(o, o, 3, 1), nn.BatchNorm2d(o),
                c(i, o, 1, s) if s != 1 else nn.Identity()]))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn0(self.stem(x)))
        for c1, b1, c2, b2, sc in self.convs:
            y = b2(c2(F.relu(b1(c1(x)))))
            x = F.relu(y + sc(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def cmd_bench(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu')
    for label, make in [('float', FloatResNet8), (f'brevitas {args.bits}',
                                                  lambda: ResNet8(parse_bits(args.bits)))]:
        model = make().to(dev).train()
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        x = torch.randn(args.batch, 3, 32, 32, device=dev)
        y = torch.randint(0, 10, (args.batch,), device=dev)
        for i in range(args.warmup + args.iters):
            if i == args.warmup:
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.iters
        print(f'{label:32s} {dt * 1000:7.1f} ms/iter (batch {args.batch}) -> '
              f'{dt * 50000 / args.batch:6.1f} s per 50k-image epoch (compute only)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['count', 'export', 'bench'])
    p.add_argument('--bits', type=int, nargs=8, default=[8] * 8)
    p.add_argument('--out', default='resnet8.onnx')
    p.add_argument('--ckpt', default=None, help='state_dict to export (export only)')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()
    {'count': cmd_count, 'export': cmd_export, 'bench': cmd_bench}[args.cmd](args)


if __name__ == '__main__':
    main()
