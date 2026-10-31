"""Stand-in for PYNQ v2.5 on a PC, for the contract test in ../run_contract.py.

buffer.py, mmio.py, registers.py, ps.py and pl_server/hwh_parser.py are PYNQ v2.5's own files
(Xilinx/PYNQ tag v2.5, BSD-3-Clause, ../PYNQ_LICENSE.txt), unchanged. Only the lowest layer is
simulated: physical memory and /dev/mem (pl_server/device.py), the CMA allocator (xlnk.py), the
bitstream download (overlay.py) and the two FINN IODMAs. So board_agent.Board and the vendored
FINN driver go through v2.5's real MMIO, Register, Clocks, PynqBuffer and hwh parser.
"""

import collections
import os

if not hasattr(os, "uname"):   # Windows; v2.5 ps.py reads os.uname().machine at import
    _uname = collections.namedtuple("uname_result", "sysname nodename release version machine")
    os.uname = lambda: _uname("Linux", "pynq", "4.19.0-xilinx-v2019.1", "#1 SMP PREEMPT", "armv7l")

__version__ = "2.5"

from .mmio import MMIO              # noqa: E402
from .xlnk import Xlnk              # noqa: E402
from .ps import Register, Clocks    # noqa: E402
from .overlay import Overlay, DefaultIP   # noqa: E402
from .buffer import allocate        # noqa: E402
