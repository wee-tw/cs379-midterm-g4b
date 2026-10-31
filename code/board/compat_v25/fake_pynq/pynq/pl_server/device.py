"""Simulated Zynq device for the PYNQ v2.5 stand-in: physical memory behind /dev/mem, the SLCR
clock registers of a PYNQ-Z2, and the two FINN IODMAs of the finn-examples shells."""

import numpy as np

PAGE = 4096
SLCR = 0xF8000000
IDMA, ODMA = 0x40000000, 0x40001000   # idma0 / odma0 in every finn-examples hwh (v2.5 parser)
MEM = {}   # page base -> uint32 words
HW = {}    # page base -> simulated peripheral


class _IODMA:
    """AXI-lite registers of a FINN IODMA: 0x00 control (ap_start bit 0, ap_done bit 1 cleared
    on read, ap_idle bit 2), 0x10 buffer address, 0x1C number of images."""

    def __init__(self, role):
        self.role = role
        self.reset()

    def reset(self):
        self.regs = {0: 0x4}
        self.runs = 0

    def window(self, off, length):
        assert off == 0, "IODMA registers are mapped from their base"
        return self

    def __getitem__(self, idx):
        v = self.regs.get(idx, 0)
        if idx == 0 and self.role == "odma":
            self.regs[0] = v & ~0x2
        return np.uint32(v)

    def __setitem__(self, idx, val):
        val = int(val)
        if idx == 0 and val & 1:
            self.regs[0] = 0   # busy
            if self.role == "idma":
                _run_accelerator()
        else:
            self.regs[idx] = val


def _run_accelerator():
    """The dataflow pipeline: one label per image (sum of its input bytes mod 10)."""
    from ..xlnk import BUFFERS
    idma, odma = HW[IDMA], HW[ODMA]
    assert odma.regs[0] == 0, "the FINN driver starts the output DMA before the input DMA"
    n = idma.regs[0x1C >> 2]
    assert odma.regs[0x1C >> 2] == n, "input and output DMA image counts differ"
    ib, ob = BUFFERS[idma.regs[0x10 >> 2]](), BUFFERS[odma.regs[0x10 >> 2]]()
    x = np.asarray(ib).reshape(ib.shape[0], -1)[:n]
    y = np.asarray(ob).reshape(ob.shape[0], -1)
    y[:n] = 0
    y[:n, 0] = (x.astype(np.int64).sum(1) % 10).astype(np.uint8)
    idma.regs[0] = odma.regs[0] = 0x6   # done and idle
    idma.runs += 1


def _init_memory():
    slcr = np.zeros(PAGE // 4, np.uint32)
    slcr[0x100 >> 2] = 26 << 12                # ARM_PLL_CTRL, FBDIV 26: 1300 MHz
    slcr[0x108 >> 2] = 20 << 12                # IO_PLL_CTRL, FBDIV 20: 1000 MHz from the 50 MHz PS_CLK
    slcr[0x170 >> 2] = (1 << 20) | (10 << 8)   # FPGA0_CLK_CTRL: IO PLL / 10 / 1 = 100 MHz
    slcr[0x17C >> 2] = 1 << 16                 # FPGA0_THR_STA: running (the throttle is not simulated)
    MEM[SLCR] = slcr
    HW[IDMA], HW[ODMA] = _IODMA("idma"), _IODMA("odma")


class _DeviceMeta(type):
    @property
    def active_device(cls):
        if Device._active is None:
            Device._active = XlnkDevice()
        return Device._active


class Device(metaclass=_DeviceMeta):
    _active = None

    def __init__(self, tag):
        self.tag = tag
        self.bitfile_name = None

    def allocate(self, shape, dtype, **kwargs):   # as v2.5 Device.allocate
        return self.default_memory.allocate(shape, dtype, **kwargs)

    def has_capability(self, cap):
        return cap in self.capabilities and self.capabilities[cap]


class XlnkDevice(Device):
    def __init__(self):
        super().__init__("xlnk")
        from ..xlnk import Xlnk
        self.default_memory = Xlnk()
        self.capabilities = {"MEMORY_MAPPED": True}

    def mmap(self, base_addr, length):   # /dev/mem
        page = base_addr & ~(PAGE - 1)
        off = base_addr - page
        assert off + length <= PAGE, "simulated mmap stays within one page"
        if page in HW:
            return HW[page].window(off, length)
        if page not in MEM:
            MEM[page] = np.zeros(PAGE // 4, np.uint32)
        return MEM[page][off >> 2:(off + length) >> 2]

    def get_bitfile_metadata(self, bitfile_name):
        from .hwh_parser import HWH, get_hwh_name
        return HWH(get_hwh_name(bitfile_name))

    def download(self, overlay):   # fpga_manager
        HW[IDMA].reset()
        HW[ODMA].reset()
        self.bitfile_name = overlay.bitfile_name


_init_memory()
