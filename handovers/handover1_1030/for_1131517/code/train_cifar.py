"""Minimal CIFAR-10 trainer for ResNet-8 (float or Brevitas), data kept on the GPU.

  python train_cifar.py --model float --epochs 40 --data cifar10.npz
  python train_cifar.py --model finnfloat --epochs 40 --data cifar10.npz
  python train_cifar.py --model brevitas --bits 4 4 4 4 4 4 8 8 --epochs 30 \
      --init finnfloat.pt --teacher float.pt --recipe mse --data cifar10.npz

Models:
  float      MLPerf Tiny ResNet-8 as published (float reference)
  finnfloat  the FINN-style topology (extra ReLUs, BN on shortcuts) in float
  brevitas   the FINN-style topology quantized with Brevitas (--bits, --recipe)

Inputs are scaled to [0, 1] (ToTensor-style, no mean/std normalisation), which is
what a FINN deployment expects: the FPGA receives raw UINT8 pixels and a single
1/255 scale is folded into the first layer. Augmentation: random crop with 4-pixel
padding and horizontal flip, done on the GPU. Images stay uint8 on the GPU so that
two trainings fit on a 6 GB card at once.

--init loads a state_dict with strict=False. --reinit-act drops the activation
quantizer scales from it, so they are re-initialised from statistics at the new
bit-width instead of inheriting the clipping range learned at another bit-width.
--teacher adds knowledge distillation from a float model's logits.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from resnet8_brevitas import ResNet8
from verify_resnet8 import FloatResNet8, parse_bits


def load_cifar(root, dev):
    """root: a torchvision CIFAR-10 folder, or an .npz with x_train/y_train/x_test/y_test
    (uint8 NHWC images), e.g. made by prepare_cifar_npz.py."""
    if root.endswith(".npz"):
        d = np.load(root)
        splits = [(d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])]
    else:
        splits = []
        for train in (True, False):
            ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
            splits.append((np.asarray(ds.data), np.asarray(ds.targets)))
    out = []
    for x, y in splits:
        out += [torch.tensor(x).permute(0, 3, 1, 2).contiguous().to(dev),
                torch.tensor(y, dtype=torch.long).to(dev)]
    return out


def to_float(x_u8):
    return x_u8.float().div_(255)


def augment(x):
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4))
    i, j = torch.randint(0, 9, (2,))
    x = x[:, :, i:i + 32, j:j + 32]  # one random offset per batch keeps it fast
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = x[flip].flip(3)
    return x


def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for k in range(0, len(x), bs):
            correct += (model(to_float(x[k:k + bs])).argmax(1) == y[k:k + bs]).sum().item()
    return 100.0 * correct / len(x)


def build(kind, bits, recipe, amid=None):
    if kind == "float":
        return FloatResNet8()
    if kind == "finnfloat":
        return ResNet8(None)
    return ResNet8(parse_bits(bits), recipe=recipe, amid=amid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["float", "finnfloat", "brevitas"], default="float")
    p.add_argument("--bits", type=int, nargs=8, default=[8] * 8)
    p.add_argument("--recipe", choices=["default", "mse", "lowbit", "lowbit_mse"], default="default")
    p.add_argument("--amid", type=int, nargs=3, default=None,
                   help="activation widths inside the residual branch of stacks 1-3")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--data", default="./data")
    p.add_argument("--init", default=None, help="state_dict to start from (QAT fine-tune)")
    p.add_argument("--reinit-act", action="store_true")
    p.add_argument("--teacher", default=None, help="float MLPerf ResNet-8 state_dict for KD")
    p.add_argument("--kd-alpha", type=float, default=0.7)
    p.add_argument("--kd-t", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    p.add_argument("--log", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    xtr, ytr, xte, yte = load_cifar(args.data, dev)
    model = build(args.model, args.bits, args.recipe, args.amid)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        if args.reinit_act:
            sd = {k: v for k, v in sd.items() if "act_quant" not in k}
        print(model.load_state_dict(sd, strict=False))
    model = model.to(dev)
    teacher = None
    if args.teacher:
        teacher = FloatResNet8()
        teacher.load_state_dict(torch.load(args.teacher, map_location="cpu"))
        teacher = teacher.to(dev).eval()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    steps = args.epochs * (len(xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.15)
    history = []
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr), device=dev)
        for k in range(0, len(xtr) - args.batch + 1, args.batch):
            idx = perm[k:k + args.batch]
            x = augment(to_float(xtr[idx]))
            out = model(x)
            loss = F.cross_entropy(out, ytr[idx])
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                T = args.kd_t
                kd = F.kl_div(F.log_softmax(out / T, 1), F.softmax(t_out / T, 1),
                              reduction="batchmean") * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        acc = evaluate(model, xte, yte)
        history.append(dict(epoch=ep + 1, sec=round(dt, 1), loss=round(loss.item(), 3),
                            test_acc=round(acc, 2)))
        print(history[-1], flush=True)
    if args.save:
        torch.save(model.state_dict(), args.save)
    if args.log:
        json.dump(dict(args=vars(args), history=history, final_acc=history[-1]["test_acc"],
                       best_acc=max(h["test_acc"] for h in history),
                       wall_sec=round(time.time() - t_start, 1)),
                  open(args.log, "w"), indent=1)


if __name__ == "__main__":
    main()
