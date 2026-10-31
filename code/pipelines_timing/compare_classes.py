"""Post-route clock class of each routed node against the class timing_safe.py gives it.

  python compare_classes.py <scratchpad> [out.csv]"""
import glob, json, os, re, sys
import timing_safe as ts
S = sys.argv[1]
rtl = ts.rtl_lookup(S + "/rtl_timing/rtl_all_constrained.json")
res = []
for f in sorted(glob.glob(S + "/route/p8_*.json")) + [S + "/route/smoke_p8_rtl.json", S + "/route/smoke_p8_hls.json"]:
    res += json.load(open(f))
def params(case):
    for root in ("D:/vw5", "D:/vt2", "D:/vrc", "D:/hf", "D:/ht"):
        d = os.path.join(root, case)
        if os.path.isdir(d):
            break
    w = glob.glob(d + "/*_wrapper.v")
    if w:
        t = open(w[0]).read()
        m = {k: re.search(rf"parameter\s+{k}\s*=\s*(\d+)", t) for k in ("MW", "MH", "PE", "SIMD", "WEIGHT_WIDTH", "ACTIVATION_WIDTH")}
        if all(m.values()):
            return ("rtl_mvu",) + tuple(int(v.group(1)) for v in m.values())
        return ("rtl_other",)
    c = glob.glob(d + "/top_*.cpp") + glob.glob(d + "/project_*/**/top_*.cpp", recursive=True)
    if c:
        t = open(c[0]).read()
        g = lambda k: int(re.search(rf"#define {k} (\d+)", t).group(1))
        return ("hls_mvu", g("MW1"), g("MH1"), g("PE1"), g("SIMD1"), g("WP1"), None)
    return ("?",)
rows, table = [], []
for r in sorted(res, key=lambda r: r["case"]):
    if r.get("route_ns") is None:
        print(f"{r['case']:38s} not routed (returncode {r.get('returncode')})")
        continue
    p = params(r["case"])
    if p[0] == "rtl_mvu":
        _, mw, mh, pe, simd, w, a = p
        k = (mw, mh, pe, simd, w, a)
        ps = rtl.get(k) or ts.rtl_model(rtl, w <= 4 and a <= 4, simd)
        cm = ts.clock_class(ps)
        desc = f"RTL MW{mw} MH{mh} PE{pe} SIMD{simd} W{w}A{a} ({'measured' if k in rtl else 'model'} {ps:.2f})"
    elif p[0] == "hls_mvu":
        _, mw, mh, pe, simd, w, _ = p
        cm = ts.clock_class(7.63) if simd == mw else (ts.clock_class(8.80) if simd <= 27 else 0.0)
        desc = f"HLS MW{mw} MH{mh} PE{pe} SIMD{simd} W{w} SF{mw // simd}"
    else:
        cm, desc = ts.clock_class(7.10), "other (SWG)"
    cr = ts.clock_class(r["route_ns"])
    verdict = "same" if cr == cm else ("model optimistic" if cm > cr else "model pessimistic")
    rows.append(verdict)
    table.append({"case": r["case"], "node": desc, "target_ns": 8.0, "post_synth_ns": r["synth_ns"],
                  "post_route_ns": r["route_ns"], "route_class_MHz": cr, "model_class_MHz": cm,
                  "verdict": verdict})
    print(f"{r['case']:38s} {desc:46s} synth {r['synth_ns']:6.3f} route {r['route_ns']:6.3f} | route {cr:g} MHz, model {cm:g} MHz: {verdict}")
print({v: rows.count(v) for v in set(rows)})
if len(sys.argv) > 2:
    import csv
    with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("wrote", sys.argv[2], len(table), "rows")
