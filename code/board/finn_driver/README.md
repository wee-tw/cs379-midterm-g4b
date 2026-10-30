# finn_driver: self-contained PYNQ driver for the finn-examples bitstreams

These are the same files that FINN v0.10.1's `MakePYNQDriver` puts into every deployment package. With them, `board_agent.py` can run on a PYNQ-Z2 without `pip install`, and without internet on the board.

| File | Source | License |
|---|---|---|
| `driver_base.py` | FINN v0.10.1 `src/finn/qnn-data/templates/driver/driver_base.py` | BSD-3-Clause (`FINN_LICENSE.txt`) |
| `finn/util/data_packing.py`, `finn/util/__init__.py` | FINN v0.10.1 `src/finn/util/` | BSD-3-Clause |
| `qonnx/core/datatype.py`, `qonnx/util/basic.py` (+ `__init__.py`) | qonnx @ fd61cfe, the version FINN v0.10.1 pins | Apache-2.0 (`QONNX_LICENSE.txt`) |
| `bitstring.py` | bitstring 3.1.9 py3 wheel from PyPI (pure Python; FINN's docs ask for `pip3 install bitstring` on the board) | MIT (`bitstring_LICENSE.txt`) |
| `io_shapes.py` | I/O shape dictionaries from finn-examples `finn_examples/models.py` | Apache-2.0 |

The bitstreams are the finn-examples v0.0.7a Pynq-Z1 release (`Pynq-Z1.zip`, md5 402ab9dfa4612ae340715bf80e4717bd). Pynq-Z1 and PYNQ-Z2 use the same xc7z020clg400-1 device, and the FINN shell uses no board pins.
