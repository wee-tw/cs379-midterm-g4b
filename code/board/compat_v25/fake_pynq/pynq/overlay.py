"""PYNQ v2.5's Overlay and DefaultIP, reduced to what the FINN driver uses: the v2.5 constructor
signature, the hwh metadata (v2.5 parser), the clock setting of v2.5 Overlay.download() and an
MMIO per IP. The bitstream itself is not loaded (pl_server/device.py)."""

import os
from copy import deepcopy

from .mmio import MMIO
from .ps import Clocks
from .pl_server.device import Device


class DefaultIP:
    def __init__(self, description):
        self.device = description.get("device") or Device.active_device
        self.mmio = MMIO(description["phys_addr"], description["addr_range"], device=self.device)

    def read(self, offset=0):
        return self.mmio.read(offset)

    def write(self, offset, value):
        self.mmio.write(offset, value)


class Overlay:
    def __init__(self, bitfile_name, dtbo=None, download=True, ignore_version=False, device=None):
        if not isinstance(bitfile_name, str):
            raise TypeError("Bitstream name has to be a string.")
        if device is None:
            device = Device.active_device
        self.device = device
        self.bitfile_name = os.path.abspath(bitfile_name)
        if not os.path.exists(self.bitfile_name):
            raise IOError("Bitstream file {} does not exist.".format(bitfile_name))
        self.parser = self.device.get_bitfile_metadata(self.bitfile_name)
        self.ip_dict = deepcopy(self.parser.ip_dict)
        self.clock_dict = self.parser.clock_dict
        self.ignore_version = ignore_version
        self._ips = {}
        if download:
            self.download()

    def download(self, dtbo=None):
        for i in self.clock_dict:   # as in v2.5 Overlay.download
            enable = self.clock_dict[i]['enable']
            div0 = self.clock_dict[i]['divisor0']
            div1 = self.clock_dict[i]['divisor1']
            if enable:
                Clocks.set_pl_clk(i, div0, div1)
            else:
                Clocks.set_pl_clk(i)
        self.device.download(self)

    def is_loaded(self):
        return self.device.bitfile_name == self.bitfile_name

    def __getattr__(self, key):
        if key.startswith("__") or "_ips" not in self.__dict__:
            raise AttributeError(key)
        if not self.is_loaded():
            raise RuntimeError("Overlay not currently loaded")
        if key not in self.ip_dict:
            raise AttributeError("Could not find IP or hierarchy {} in overlay".format(key))
        if key not in self._ips:
            self._ips[key] = DefaultIP(dict(self.ip_dict[key], device=self.device))
        return self._ips[key]
