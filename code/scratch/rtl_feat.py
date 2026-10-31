"""Which features predict the Vivado LUTs of an MVAU_rtl unit? Audit units + design units."""
import glob, json, os, re, sys
import numpy as np
S = sys.argv[1]
def params(d):
    w = glob.glob(os.path.join(d, "*_wrapper.v"))
    w = [x for x in w if not x.endswith("_sim.v")][0]
    t = open(w).read()
    p = {k: v for k, v in re.findall(r"parameter\s+(\w+)\s*=\s*\"?(\w+)\"?", t)}
    return p
rows = []
for c in json.load(open(f"{S}/audit/audit.json")):
    if c["op"] == "MVAU_rtl" and c["vivado"].get("returncode") == 0 and "bracket" in c:
        p = params(f"{S}/audit/{c['case']}")
        rows.append((p, c["vivado"]["LUT"], "audit", c["bracket"]["hi"]))
for dd in ("D:/vd", "D:/vd2"):
    if not os.path.exists(f"{dd}/results.json"):
        continue
    res = json.load(open(f"{dd}/results.json"))
    for c in json.load(open(f"{dd}/cases.json")):
        if c["op"] == "MVAU_rtl" and c["kind"] != "wstrm":
            v = res.get(c["case"], {})
            if v.get("returncode") == 0 and v.get("LUT") is not None:
                rows.append((params(f"{dd}/{c['case']}"), v["LUT"], f"{os.path.basename(dd)}:{c['design']}", None))
print(len(rows), "units;", {r[2] for r in rows})
print("COMPUTE_CORE values:", sorted({r[0].get("COMPUTE_CORE") for r in rows}))
I = lambda r, k: float(r[0][k])
y = np.array([r[1] for r in rows]); g = [r[2] for r in rows]
PE = np.array([I(r, "PE") for r in rows]); SI = np.array([I(r, "SIMD") for r in rows])
A = np.array([I(r, "ACTIVATION_WIDTH") for r in rows]); W = np.array([I(r, "WEIGHT_WIDTH") for r in rows])
ACC = np.array([I(r, "ACCU_WIDTH") for r in rows]); MW = np.array([I(r, "MW") for r in rows]); MH = np.array([I(r, "MH") for r in rows])
one = np.ones_like(y); L = PE * SI
models = {
    "lanes": np.c_[L, one],
    "lanes,PE*ACC,SIMD*A": np.c_[L, PE * ACC, SI * A, one],
    "lanes,PE,SIMD": np.c_[L, PE, SI, one],
    "lanes*W,lanes*A": np.c_[L * W, L * A, one],
    "lanes,PE*ACC": np.c_[L, PE * ACC, one],
    "lanes,SIMD*A": np.c_[L, SI * A, one],
    "lanes,lanes*W": np.c_[L, L * W, one],
}
def lst(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
designs = sorted(set(g) - {"audit"})
for name, X in models.items():
    loo = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        loo.append(abs(X[i] @ lst(X[m], y[m]) - y[i]) / max(y[i], 1))
    lodo = []
    for d in designs:
        te = np.array([x == d for x in g])
        p = X[te] @ lst(X[~te], y[~te])
        lodo.append(f"{100 * (p.sum() / y[te].sum() - 1):+.0f}%({te.sum()})")
    aud = np.array([x == "audit" for x in g])
    print(f"{name:22s} coef {np.round(lst(X, y), 2)} LOO med {100 * np.median(loo):.0f}% p90 {100 * np.percentile(loo, 90):.0f}% | LODO {' '.join(lodo)}")
# residuals of the lanes model on design units
c = lst(models["lanes"], y)
print("design units, lanes model: case PE SIMD W A ACC synth pred")
for r, yy, gg, p, l in zip(rows, y, g, models["lanes"] @ c, L):
    if gg != "audit":
        q = r[0]
        print(f"  {gg[-22:]:22s} PE {q['PE']:>3} SIMD {q['SIMD']:>4} W {q['WEIGHT_WIDTH']} A {q['ACTIVATION_WIDTH']} ACC {q['ACCU_WIDTH']:>2} synth {yy:6.0f} pred {p:6.0f}")
