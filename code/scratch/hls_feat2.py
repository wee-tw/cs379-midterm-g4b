"""HLS MVU LUT model with FINN's formula split into compute and threshold parts (TMEM = 1: constants)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, ".")
import calib_v2 as cv
S = sys.argv[1]
bits = lambda s: int("".join(ch for ch in s if ch.isdigit()))

def parts(MW, MH, PE, SIMD, W, A, B, thr, res):
    """FINN v0.10.1 MVAU lut_estimation terms (x1.1, per design): compute, threshold, by TMEM class."""
    mult = 0 if res == "dsp" else SIMD * (2 * math.ceil((W + A) / 6) - 1) * (W + A)
    adder = (W + A) * (2 * SIMD - 1)
    comp = 1.1 * PE * (mult + adder)
    tmem = MH // PE
    nthr = (2 ** B - 1) * (1 + math.ceil(tmem / 64)) * PE * 1.1 if thr else 0.0
    return comp, nthr

rows = []
for a in (f"{S}/audit_hls_finn", f"{S}/audit_hls"):
    by = {}
    for c in json.load(open(os.path.join(a, "audit_hls.json"))):
        by.setdefault(c["case"].rsplit("_", 1)[0], {})[c["style"]] = c
    seen = set()
    for v in by.values():
        if not all((v.get(s, {}).get("synth") or {}).get("LUT") for s in ("auto", "distributed")):
            continue
        at = v["auto"]["attrs"]
        key = (tuple(sorted(at.items())), v["auto"]["synth"]["LUT"])
        if key in seen:
            continue
        seen.add(key)
        MW, MH, PE, SIMD = (int(at[k]) for k in ("MW", "MH", "PE", "SIMD"))
        W, A, B = bits(at["weightDataType"]), bits(at["inputDataType"]), bits(at["outputDataType"])
        comp, nthr = parts(MW, MH, PE, SIMD, W, A, B, True, "lut")
        rows.append(dict(g="audit", MW=MW, MH=MH, PE=PE, SIMD=SIMD, W=W, A=A, comp=comp, nthr=nthr,
                         cons=v["distributed"]["finn_estimate"]["LUT"], y=v["auto"]["synth"]["LUT"], res="lut"))
for dd in ("D:/vd", "D:/vd2"):
    tag_bits = {}
    for n in cv.design_nodes([dd], f"{S}/cost/tables_calov.json"):
        if n["op"] != "MVAU_hls" or n["dsp_mult"] or "LUT_cons" not in n["opt"]:
            continue
        sh, fo = n["node"]["shape"], n["node"]["fold"]
        res = n["opt"].get("attrs", {}).get("resType", "lut")
        thr = any("ram_style_thresholds" in v.get("attrs", {}) for v in n["node"].get("variants", []))
        # output bits: this group's activation width (the tables key "a"); the final FC has no thresholds
        tag = n["design"].split(":")[1]; b = list(map(int, tag.split("_")[0]))
        B = b[2 * n["node"]["group"] + 1] if n["node"]["group"] < 4 else 0
        comp, nthr = parts(sh["MW"], sh["MH"], fo["PE"], fo["SIMD"], sh["W"], sh["A"], B, thr, res)
        rows.append(dict(g=n["design"], MW=sh["MW"], MH=sh["MH"], PE=fo["PE"], SIMD=fo["SIMD"], W=sh["W"], A=sh["A"],
                         comp=comp, nthr=nthr, cons=cv.split(n)[0], y=n["synth"], res=res))
print(len(rows), "units;", sum(r["g"] == "audit" for r in rows), "audit")
y = np.array([r["y"] for r in rows], float); g = [r["g"] for r in rows]
f = lambda k: np.array([r[k] for r in rows], float)
SF = f("MW") / f("SIMD"); NF = f("MH") / f("PE")
buf = f("SIMD") * f("A") * ((SF > 1) & (NF > 1))
lnw = f("PE") * f("SIMD") * f("W") * np.array([r["res"] == "lut" for r in rows])
t1 = f("nthr") * (NF == 1); tn = f("nthr") * (NF > 1)
one = np.ones_like(y)
P_, Q_, W_, A_ = f("PE"), f("SIMD"), f("W"), f("A")
lutm = np.array([r["res"] == "lut" for r in rows], float)
Bthr = f("nthr") / np.maximum(1e-9, 1.1 * P_ * (1 + np.ceil(f("MH") / f("PE") / 64)))  # = 2^B - 1 (0: no thresholds)
th1 = P_ * Bthr * (NF == 1); thn = P_ * Bthr * (NF > 1)
x_mul = P_ * Q_ * W_ * A_ * lutm; x_add = P_ * Q_ * (W_ + A_)
models = {
    "v2: cons, buf, lnw": np.c_[f("cons"), buf, lnw],
    "v2 + thr[TMEM=1]": np.c_[f("cons"), buf, lnw, th1],
    "v2 + thr[TMEM=1] + thr[TMEM>1]": np.c_[f("cons"), buf, lnw, th1, thn],
    "v2 + PE[TMEM=1]": np.c_[f("cons"), buf, lnw, P_ * (NF == 1)],
}
lst = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
designs = sorted(set(g) - {"audit"})
print("designs:", [d.split(":")[0] + ":" + d.split(":")[1][:14] for d in designs])
for name, X in models.items():
    n = len(y)
    loo = [abs(X[i] @ lst(X[np.arange(n) != i], y[np.arange(n) != i]) - y[i]) / y[i] for i in range(n)]
    lodo = []
    for d in designs:
        te = np.array([x == d for x in g]); p = X[te] @ lst(X[~te], y[~te])
        lodo.append(f"{100 * (p.sum() / y[te].sum() - 1):+.0f}%")
    print(f"{name:46s} coef {np.round(lst(X, y), 3)} LOO med {100 * np.median(loo):.0f}% p90 {100 * np.percentile(loo, 90):.0f}% | LODO {' '.join(lodo)}")
