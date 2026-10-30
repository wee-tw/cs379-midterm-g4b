"""Convert the Hugging Face copy of CIFAR-10 (uoft-cs/cifar10, parquet) into cifar10.npz.

The official Toronto tarball downloaded very slowly here, so we use the HF mirror:
  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet

  python prepare_cifar_npz.py <folder with train.parquet and test.parquet>

Writes <folder>/cifar10.npz with uint8 NHWC x_train/x_test and int64 y_train/y_test.
"""

import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main(folder):
    out = {}
    for split in ("train", "test"):
        t = pq.read_table(os.path.join(folder, split + ".parquet")).to_pydict()
        out["x_" + split] = np.stack(
            [np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        out["y_" + split] = np.array(t["label"], dtype=np.int64)
        print(split, out["x_" + split].shape, np.bincount(out["y_" + split]))
    np.savez(os.path.join(folder, "cifar10.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1])
