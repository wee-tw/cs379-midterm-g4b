"""Restore the JSON cost tables from the .json.gz files (GitHub keeps only the compressed copies).

The .npz point files need no unpacking: np.load reads them as they are.

  python code/cost/unpack.py                          # every table, about 2.5 GB
  python code/cost/unpack.py tables_calv2_tsr_v1218   # only the files whose name starts with this
"""

import glob
import gzip
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    prefixes = sys.argv[1:]
    for gz in sorted(glob.glob(os.path.join(HERE, "**", "*.json.gz"), recursive=True)):
        name = os.path.basename(gz)
        if prefixes and not any(name.startswith(p) for p in prefixes):
            continue
        out = gz[:-3]
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(gz):
            continue
        with gzip.open(gz, "rb") as fi, open(out + ".tmp", "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        os.replace(out + ".tmp", out)
        print(os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
