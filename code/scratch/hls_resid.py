import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, ".")
import calib_v2 as cv
S = sys.argv[1]
def hparams(d):
    t = open(glob.glob(os.path.join(d, "top_*.cpp"))[0]).read()
    g = lambda k: int(re.search(rf"#define {k}1 (\d+)", t).group(1))
    m = re.search(r"Matrix_Vector_Activate_Stream_Batch<\w+, \w+, \w+, \w+, Slice<ap_(u?)int<(\d+)>>, Slice<ap_(u?)int<(\d+)>>, (\w+), ap_(u?)int<(\d+)> >", t)
    res = re.search(r"ap_resource_(\w+)\(\)", t).group(1)
    th = open(os.path.join(d, "thresh.h")).read() if os.path.exists(os.path.join(d, "thresh.h")) else ""
    mt = re.search(r"ThresholdsActivation<(\d+),(\d+),(\d+),ap_(u?)int<(\d+)>", th)
    return dict(MW=g("MW"), MH=g("MH"), SIMD=g("SIMD"), PE=g("PE"), A=int(m.group(2)), O=int(m.group(4)),
                W=int(m.group(7)), res=res, steps=int(mt.group(3)) if mt else 0, acc=int(mt.group(5)) if mt else 0,
                tmem=int(mt.group(1)) if mt else 0)
dn = cv.design_nodes(sys.argv[2:], f"{S}/cost/tables_calov.json")
f = json.load(open(f"{S}/cost/calib_v2.json"))
print("design            case                      PE SIMD  W A acc st res | cons   wl   pred  synth  ratio")
for n in dn:
    if n["op"] != "MVAU_hls" or n["dsp_mult"] or "LUT_cons" not in n["opt"]:
        continue
    case = [c for c in json.load(open(os.path.join("D:/" + n["design"].split(":")[0], "cases.json")))]
    h, wl = cv.split(n)
    p = f["k_hls"] * h
    # find case dir by matching synth value is fragile; use node name + design
    dd = "D:/" + n["design"].split(":")[0]
    tag = n["design"].split(":")[1]
    cs = [c for c in case if c["design"] == tag and c["case"].endswith(n["node"]["name"]) and c["kind"] != "wstrm"]
    q = hparams(os.path.join(dd, cs[0]["case"]))
    print(f"{tag[:17]:17s} {cs[0]['case']:24s} {q['PE']:3d} {q['SIMD']:4d} {q['W']:2d} {q['A']} {q['acc']:3d} {q['steps']:2d} {q['res'][:3]} | {h:6.0f} {wl:5.0f} {p:6.0f} {n['synth']:6.0f} {n['synth']/max(p,1):5.2f}")
