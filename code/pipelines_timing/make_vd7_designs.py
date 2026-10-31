"""B2 designs (FINN default folding, resType lut, 100 MHz) of every margin, to get their Vivado
dynamic power next to the A' designs (notes 12.43): the measured B2 re-pick when it exists.
Margin 1 (44444448) was synthesized before (verify_b2lut) and is skipped.
  make_vd7_designs.py <main table json> <points f100 npz> <out designs json>"""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:/115-1/embedded sys midterm report/code")
import energy_search as es  # noqa: E402

t = json.load(open(sys.argv[1]))
rows = np.load(sys.argv[2])["rows"]
idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
out, seen = [], {"44444448"}
for m in sorted(t, key=float):
    r = t[m]["nominal"]
    b = (r.get("B2_measured") or {}).get("bits") or r["B2_proxy"]["bits"]
    if b in seen:
        continue
    seen.add(b)
    mask = (rows[:, 0].astype(int) == idx[b]) & (rows[:, 1] == es.MODES.index("lut_finn"))
    i = int(np.argmin(np.where(mask, rows[:, 3], np.inf)))   # the fastest design, as B2 is picked
    out.append([b, "lut_finn", int(rows[i, 2]), 100.0, f"B2 m{float(m):g} (resType lut)",
                {"MVAU_hls": {"resType": "lut"}}])
json.dump(out, open(sys.argv[3], "w"), indent=1)
for d in out:
    print(d)
