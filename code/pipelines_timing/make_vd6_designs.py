"""Designs to synthesize node by node after the routing-checked refold (RTL <= 12, HLS <= 18):
the verified A' and the B1 design of every margin in main_table_tsr_v1218_m0.json (nominal).
  make_vd6_designs.py <main table json> <out designs json>"""
import json
import sys

t = json.load(open(sys.argv[1]))
out, seen = [], set()
for m in sorted(t, key=float):
    r = t[m]["nominal"]
    a = r["A_verified"]
    picks = [(a["bits"], a["mode"], a["T_target"], a["f_MHz"], f"A' m{float(m):g}")]
    b1 = r["B1_uniform"]
    b1m = r.get("B1_measured", {})
    if b1m and b1m["bits"] == b1["bits"]:
        picks.append((b1["bits"], b1["mode"], b1["T_target"], b1["f_MHz"], f"B1 m{float(m):g}"))
    for p in picks:
        key = p[:3]
        if key in seen:
            continue
        seen.add(key)
        out.append(list(p))
json.dump(out, open(sys.argv[2], "w"), indent=1)
for d in out:
    print(d)
