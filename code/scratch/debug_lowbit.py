"""Why does the 'lowbit' recipe kill W2A2? Train a few steps and watch the quantizers."""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\115-1\embedded sys midterm report\code")
from brevitas.quant_tensor import QuantTensor  # noqa: E402
from resnet8_brevitas import ResNet8  # noqa: E402
from train_cifar import augment, to_float  # noqa: E402
from verify_resnet8 import parse_bits  # noqa: E402

recipe = sys.argv[1]
init, npz = sys.argv[2], sys.argv[3]
dev = torch.device("cuda")
d = np.load(npz)
x = torch.tensor(d["x_train"][:4096]).permute(0, 3, 1, 2).contiguous().to(dev)
y = torch.tensor(d["y_train"][:4096]).to(dev)
m = ResNet8(parse_bits([2, 2, 2, 2, 2, 2, 8, 8]), recipe=recipe)
print(m.load_state_dict(torch.load(init, map_location="cpu"), strict=False).missing_keys[:3])
m = m.to(dev).train()
opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)

stats = {}


def hook(name):
    def f(mod, inp, out):
        v = out.value if isinstance(out, QuantTensor) else out
        stats[name] = (float((v == 0).float().mean()), float(v.max()),
                       float(out.scale.flatten()[0]) if isinstance(out, QuantTensor) else None)
    return f


for name in ["stem_act", "stack1.relu1", "stack1.relu_out", "stack2.relu1", "stack2.relu_out",
             "stack3.relu1", "stack3.relu_out"]:
    dict(m.named_modules())[name].register_forward_hook(hook(name))

for step in range(400):
    idx = torch.randint(0, 4096, (128,), device=dev)
    out = m(augment(to_float(x[idx])))
    loss = F.cross_entropy(out, y[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in (0, 10, 100, 299, 301, 399):
        ws = m.stack2.conv1.quant_weight()
        print(f"step {step} loss {loss.item():.3f} logits std {out.std().item():.3f} "
              f"w2 zero frac {(ws.int() == 0).float().mean().item():.2f} "
              f"w2 scale {ws.scale.flatten()[0].item():.4f}")
        for k, v in stats.items():
            print(f"   {k:16s} zero {v[0]:.2f} max {v[1]:.3f} scale {v[2]}")
