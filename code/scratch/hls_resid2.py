"""v2 HLS model residuals per unit: audits, D:/vd designs (in the fit) and D:/vd2 (held out)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import calib_v2 as cv
S = sys.argv[1]
f = json.load(open(f"{S}/cost/calib_v2.json"))
_, hls_a, _ = cv.audit_points(S)
print("AUDIT units (all in the fit): cons buf lnw -> pred synth ratio")
rows = []
for x, y in hls_a:
    p = float(np.dot(f["c_hls"], x)); rows.append(("audit", x, y, p))
for dd in ("D:/vd", "D:/vd2"):
    for n in cv.design_nodes([dd], f"{S}/cost/tables_calov.json"):
        if n["op"] == "MVAU_hls" and "LUT_cons" in n["opt"] and not n["dsp_mult"]:
            x = cv.node_hls_features(n); p = float(np.dot(f["c_hls"], x))
            fo, sh = n["node"]["fold"], n["node"]["shape"]
            rows.append((n["design"], x, n["synth"], p, fo, sh))
for r in rows:
    tag = r[0]
    extra = ""
    if len(r) > 4:
        fo, sh = r[4], r[5]
        extra = f"PE {fo['PE']:2d} SIMD {fo['SIMD']:3d} MW {sh['MW']:3d} MH {sh['MH']:2d} W {sh['W']} A {sh['A']}"
    print(f"{tag[-26:]:26s} {extra:44s} cons {r[1][0]:7.0f} buf {r[1][1]:5.0f} lnw {r[1][2]:5.0f} | pred {r[3]:6.0f} synth {r[2]:6.0f} {r[2]/r[3]:5.2f}")
