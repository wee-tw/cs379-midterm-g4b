"""HLS MVU compute LUTs: does an input-buffer term explain the under-prediction at SF > 1?"""
import glob, json, math, os, re, sys
import numpy as np
sys.path.insert(0, ".")
import calib_v2 as cv
from calib_mvau_hls import load_points
S = sys.argv[1]
bits = lambda s: int(re.search(r"(\d+)$", s).group(1))
def hparams(d):
    t = open(glob.glob(os.path.join(d, "top_*.cpp"))[0]).read()
    g = lambda k: int(re.search(rf"#define {k}1 (\d+)", t).group(1))
    m = re.search(r"Slice<ap_u?int<(\d+)>>", t)
    w = re.search(r"Identity, ap_u?int<(\d+)> >", t) or re.search(r", ap_u?int<(\d+)> >\s*\(", t)
    return dict(MW=g("MW"), MH=g("MH"), SIMD=g("SIMD"), PE=g("PE"), A=int(m.group(1)), W=int(w.group(1)))
rows = []
# audit points: (consistent, synth) of the auto style, keyed back to attrs
for a in (f"{S}/audit_hls_finn", f"{S}/audit_hls"):
    by = {}
    for c in json.load(open(os.path.join(a, "audit_hls.json"))):
        by.setdefault(c["case"].rsplit("_", 1)[0], {})[c["style"]] = c
    for v in by.values():
        if not all((v.get(s, {}).get("synth") or {}).get("LUT") for s in ("auto", "distributed")):
            continue
        at = v["auto"]["attrs"]
        rows.append(dict(MW=int(at["MW"]), MH=int(at["MH"]), SIMD=int(at["SIMD"]), PE=int(at["PE"]),
                         A=bits(at["inputDataType"]), W=bits(at["weightDataType"]), cons=v["distributed"]["finn_estimate"]["LUT"],
                         y=v["auto"]["synth"]["LUT"], g="audit", key=(tuple(sorted(at.items())), v["auto"]["synth"]["LUT"])))
seen, uniq = set(), []
for r in rows:
    if r["key"] not in seen:
        seen.add(r["key"]); uniq.append(r)
rows = uniq
for dd in sys.argv[2:]:
    dn = cv.design_nodes([dd], f"{S}/cost/tables_calov.json")
    cases = json.load(open(os.path.join(dd, "cases.json")))
    for n in dn:
        if n["op"] != "MVAU_hls" or n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            continue
        q = hparams(n["case"])
        q.update(cons=cv.split(n)[0], y=n["synth"], g=n["design"])
        rows.append(q)
print(len(rows), "units", sum(r["g"] == "audit" for r in rows), "audit")
y = np.array([r["y"] for r in rows], float); g = [r["g"] for r in rows]
cons = np.array([r["cons"] for r in rows], float)
SF = np.array([r["MW"] / r["SIMD"] for r in rows]); NF = np.array([r["MH"] / r["PE"] for r in rows])
SA = np.array([r["SIMD"] * r["A"] for r in rows], float)
buf = (SF > 1) & (NF > 1)
LN = np.array([r["SIMD"] * r["PE"] for r in rows], float); WB = np.array([r["W"] for r in rows], float)
models = {
    "k*cons": np.c_[cons],
    "k*cons + c*SIMD*A[SF>1]": np.c_[cons, SA * buf],
    "k*cons + c*SIMD*A*ceil(SF/4)[SF>1]": np.c_[cons, SA * np.ceil(SF / 4) * buf],
    "k*cons + c*MW*A[SF>1]": np.c_[cons, SF * SA * buf],
    "k*cons + c*SIMD*A[SF>1] + b": np.c_[cons, SA * buf, np.ones_like(y)],
    "k*cons + c*SA[SF>1] + d*lanes": np.c_[cons, SA * buf, LN],
    "k*cons + c*SA[SF>1] + d*lanes*W": np.c_[cons, SA * buf, LN * WB],
    "k*cons + c*SA[SF>1] + d*lanes[SF=1]": np.c_[cons, SA * buf, LN * (SF == 1)],
    "k*cons[SF=1] + k2*cons[SF>1] + c*SA[SF>1]": np.c_[cons * (SF == 1), cons * (SF > 1), SA * buf],
}
lst = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
designs = sorted(set(g) - {"audit"})
for name, X in models.items():
    loo = [abs(X[i] @ lst(X[np.arange(len(y)) != i], y[np.arange(len(y)) != i]) - y[i]) / y[i] for i in range(len(y))]
    lodo = []
    for d in designs:
        te = np.array([x == d for x in g]); p = X[te] @ lst(X[~te], y[~te])
        lodo.append(f"{100 * (p.sum() / y[te].sum() - 1):+.0f}%({te.sum()})")
    print(f"{name:36s} coef {np.round(lst(X, y), 3)} LOO med {100 * np.median(loo):.0f}% p90 {100 * np.percentile(loo, 90):.0f}% | LODO {' '.join(lodo)}")
X = models["k*cons + c*SIMD*A[SF>1]"]; c = lst(X, y)
print("audit units with SF>1:", int((buf & np.array([x == 'audit' for x in g])).sum()))
for r, p, yy in zip(rows, X @ c, y):
    if r["g"] != "audit":
        print(f"  {r['g'][-24:]:24s} W {r['W']} MW {r['MW']:3d} MH {r['MH']:2d} PE {r['PE']} SIMD {r['SIMD']:3d} A {r['A']} SF {r['MW']/r['SIMD']:.0f} | pred {p:6.0f} synth {yy:6.0f} {yy/p:5.2f}")
