"""Placeholder: v2.5 mmio.py imports tinynumpy, which is only used for Alveo (REGISTER_RW) devices."""


class ndarray:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("tinynumpy is not simulated")
