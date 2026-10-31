"""PYNQ v2.5's CMA allocator (/dev/xlnk + libcma), simulated: cma_array() builds the same
PynqBuffer as v2.5 (device_address, coherent = not cacheable, bo, device = this Xlnk)."""

import functools
import numbers
import os
import weakref

import numpy as np

from .buffer import PynqBuffer

BUFFERS = {}   # device address -> weak reference to the buffer (read by the simulated IODMAs)


class Xlnk:
    _next = 0x1F000000
    flushes = invalidates = 0

    def __init__(self):
        if hasattr(os, "getuid") and os.getuid() != 0:
            raise RuntimeError("Root permission needed by the library.")

    def allocate(self, *args, **kwargs):
        return self.cma_array(*args, **kwargs)

    def cma_array(self, shape, dtype=np.uint32, cacheable=0, pointer=None, cache=None):
        if isinstance(shape, numbers.Integral):
            shape = [shape]
        dtype = np.dtype(dtype)
        elements = functools.reduce(lambda value, total: value * total, shape)
        length = elements * dtype.itemsize
        buffer = bytearray(length)              # cma_alloc + cma_get_buffer
        physical_address = Xlnk._next           # cma_get_phy_addr
        Xlnk._next += (length + 0xFFF) & ~0xFFF
        view = PynqBuffer(shape=shape, dtype=dtype, buffer=buffer,
                          device_address=physical_address,
                          coherent=not cacheable,
                          bo=physical_address, device=self)
        view.pointer = 0
        view.return_to = cache
        BUFFERS[physical_address] = weakref.ref(view)
        return view

    @staticmethod
    def cma_mem_size(_=None):   # the v2.5 hwh parser calls Xlnk.cma_mem_size(None)
        return 128 << 20

    def flush(self, bo, offset, vaddr, nbytes):
        Xlnk.flushes += 1

    def invalidate(self, bo, offset, vaddr, nbytes):
        Xlnk.invalidates += 1
