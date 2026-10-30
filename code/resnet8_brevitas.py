"""MLPerf Tiny ResNet-8 (classic 16/32/64 version) in Brevitas, with per-block bit-widths.

Topology follows mlcommons/tiny benchmark/training/image_classification/keras_model.py
(2021 version): stem 3x3x16, three stacks (16, 32, 64), 1x1 stride-2 shortcuts in
stacks 2 and 3, 8x8 average pool, dense 10.

FINN-related changes, following brevitas_examples/bnn_pynq/models/resnet.py at the
Brevitas commit pinned by FINN v0.10.1 (84f42259):
  * Both operands of every residual add pass through the *same* QuantReLU instance,
    so they have identical scale and datatype (FINN's InferAddStreamsLayer skips an
    Add whose two input datatypes differ).
  * Hence the residual branch gets a ReLU before the add, and the 1x1 shortcut gets
    a BatchNorm + the shared QuantReLU. The MLPerf Keras model has neither.
  * No input quantizer: the input is assumed to be 8-bit RGB, as in the template.

Bit-width groups (see docs/midterm_plan_A_prime.md, section 5.3):
  G1 = stem + stack 1, G2 = stack 2, G3 = stack 3, G4 = FC.
  For G4, "A" is the bit-width of the pooled features fed to the FC layer.

bits=None builds the same FINN-style topology in plain float (same parameter names),
used as the starting point for QAT.

recipe selects the quantizer scale initialisation:
  "default":    max-abs per-channel weight scale, 99.999th-percentile activation init
                (Brevitas defaults, as in the bnn_pynq ResNet example)
  "mse":        MSE-optimal weight scale every step (slow: ~7x per epoch) and
                MSE-initialised activation scale
  "lowbit":     for <=4-bit weights a learned per-channel scale initialised at
                mean|w| + k*std|w| (k = 0.5 / 2 / 3 for 2 / 3 / 4 bits; with max-abs,
                2-bit ternary weights are ~83% zeros), max-abs above 4 bits; activation
                threshold initialised at a bit-dependent percentile (99 / 99.9 / 99.99 /
                99.999 for 2 / 3 / 4 / >4 bits)
  "lowbit_mse": "lowbit" weights with MSE-initialised activations
"""

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerChannelFloatMSE
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloatMSE
from brevitas.quant import IntBias
from brevitas.quant import TruncTo8bit
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMSE
from brevitas.quant_tensor import QuantTensor

WEIGHT_SIGMA = {2: 0.5, 3: 2.0, 4: 3.0}
ACT_PERCENTILE = {2: 99.0, 3: 99.9, 4: 99.99}


def weight_quant(recipe, bits, per_channel=True):
    base = Int8WeightPerChannelFloat if per_channel else Int8WeightPerTensorFloat
    if recipe == "mse":
        return Int8WeightPerChannelFloatMSE if per_channel else Int8WeightPerTensorFloatMSE
    if recipe in ("lowbit", "lowbit_mse") and bits in WEIGHT_SIGMA:
        return base.let(scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                        scaling_stats_op=StatsOp.MEAN_SIGMA_STD, sigma=WEIGHT_SIGMA[bits])
    return base


def act_quant(recipe, bits):
    if recipe in ("mse", "lowbit_mse"):
        return Uint8ActPerTensorFloatMSE
    if recipe == "lowbit" and bits in ACT_PERCENTILE:
        return Uint8ActPerTensorFloat.let(high_percentile_q=ACT_PERCENTILE[bits])
    return Uint8ActPerTensorFloat


class Builder:
    """Creates conv / ReLU layers, quantized or float, for one recipe."""

    def __init__(self, recipe):
        self.quantized = recipe is not None
        self.recipe = recipe

    def conv(self, cin, cout, k, stride, w_bits):
        if not self.quantized:
            return nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        return qnn.QuantConv2d(
            cin, cout, kernel_size=k, stride=stride, padding=k // 2, bias=False,
            weight_quant=weight_quant(self.recipe, w_bits), weight_bit_width=w_bits)

    def relu(self, a_bits):
        if not self.quantized:
            return nn.ReLU()
        return qnn.QuantReLU(act_quant=act_quant(self.recipe, a_bits), bit_width=a_bits,
                             return_quant_tensor=True)


class Stack(nn.Module):
    """One MLPerf ResNet-8 stack: conv-BN-ReLU, conv-BN, (shortcut), add, ReLU."""

    def __init__(self, b, cin, cout, stride, w_bits, a_bits, shared_act, a_mid=None):
        super().__init__()
        self.quantized = b.quantized
        self.conv1 = b.conv(cin, cout, 3, stride, w_bits)
        self.bn1 = nn.BatchNorm2d(cout)
        # relu1 sits inside the residual branch, so FINN lets it differ from the
        # residual-stream width a_bits (only the two add inputs must match)
        self.relu1 = b.relu(a_bits if a_mid is None else a_mid)
        self.conv2 = b.conv(cout, cout, 3, 1, w_bits)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            # The shortcut gets its own quantizer, shared with the residual branch.
            self.shortcut = nn.Sequential(
                b.conv(cin, cout, 1, stride, w_bits), nn.BatchNorm2d(cout), b.relu(a_bits))
            self.add_act = self.shortcut[-1]
        else:
            # Identity shortcut: x is already quantized by shared_act upstream,
            # so the residual branch must reuse exactly that quantizer.
            self.shortcut = None
            self.add_act = shared_act
        self.relu_out = b.relu(a_bits)

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.add_act(self.bn2(self.conv2(y)))
        s = x if self.shortcut is None else self.shortcut(x)
        if self.quantized:
            assert isinstance(y, QuantTensor) and isinstance(s, QuantTensor)
        return self.relu_out(y + s)


class ResNet8(nn.Module):

    def __init__(self, bits=((8, 8), (8, 8), (8, 8), (8, 8)), num_classes=10, recipe="default",
                 amid=None):
        """bits: ((W, A) for G1, G2, G3, G4), or None for the float FINN-style model.
        amid: optional activation widths inside the residual branch of stacks 1-3
        (relu1); by default they equal the stack's A."""
        super().__init__()
        b = Builder(None if bits is None else recipe)
        (w1, a1), (w2, a2), (w3, a3), (w4, a4) = bits or ((None, None),) * 4
        self.bits = bits
        self.stem = b.conv(3, 16, 3, 1, w1)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_act = b.relu(a1)
        am = amid or (None, None, None)
        self.stack1 = Stack(b, 16, 16, 1, w1, a1, shared_act=self.stem_act, a_mid=am[0])
        self.stack2 = Stack(b, 16, 32, 2, w2, a2, shared_act=None, a_mid=am[1])
        self.stack3 = Stack(b, 32, 64, 2, w3, a3, shared_act=None, a_mid=am[2])
        if b.quantized:
            # FLOOR truncation is the variant FINN supports (as in the template).
            self.pool = qnn.TruncAvgPool2d(
                kernel_size=8, trunc_quant=TruncTo8bit.let(bit_width=a4),
                float_to_int_impl_type='FLOOR')
            self.fc = qnn.QuantLinear(
                64, num_classes, bias=True,
                weight_quant=weight_quant(recipe, w4, per_channel=False),
                weight_bit_width=w4, bias_quant=IntBias)
        else:
            self.pool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem(x)))
        x = self.stack3(self.stack2(self.stack1(x)))
        x = self.pool(x)
        if isinstance(x, QuantTensor):
            # TruncAvgPool2d returns sum / 2^shift with shift = A3 + 6 - A4, i.e. the true
            # average times 2^(A4 - A3) (64x for A3=2, A4=8). Rescale so the FC always sees
            # the average, whatever the bit-widths; FINN absorbs this positive scalar into
            # the final TopK. Without it, QAT from a float model (or from another bit
            # configuration) starts with logits off by up to 64x and diverges.
            c = 2.0 ** (self.bits[2][1] - self.bits[3][1])
            x = QuantTensor(x.value * c, x.scale * c, x.zero_point, x.bit_width, x.signed,
                            x.training)
        return self.fc(x.flatten(1))


def count_weights_and_macs(model, input_shape=(1, 3, 32, 32)):
    """Weights (conv + FC, no bias/BN) and MACs per layer, from a forward pass."""
    rows = []

    def hook(m, inp, out):
        out = out.value if isinstance(out, QuantTensor) else out
        w = m.weight.numel()
        if isinstance(m, nn.Conv2d):
            macs = out.shape[2] * out.shape[3] * w
        else:
            macs = w
        rows.append((m.name, w, macs))

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.name = name
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return rows
