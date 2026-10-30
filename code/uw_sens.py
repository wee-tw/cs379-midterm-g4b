import contextlib, io, json, sys, types
sys.path.insert(0, ".")
import numpy as np
import energy_search as es
S = sys.argv[1]
for w in (0.1, 0.25, 0.5, 1.0):
    es.UTIL_W.update({"BRAM_18K": w, "DSP": w})
    out = S + f"/cost/idle_uw{w}.json"
    with contextlib.redirect_stdout(io.StringIO()):
        es.cmd_idle(types.SimpleNamespace(points=S + "/cost/points.npz", acc=S + "/cost/acc_model.json", out=out))
    d = json.load(open(out))
    for p in ("nominal", "z2_anchor"):
        c = d[p]["continuous"]
        g = [100 * (d[p][R]["gated"]["B4_latency_mJ"] / d[p][R]["gated"]["A_prime"]["E_mJ"] - 1) for R in ("30", "300", "3000")]
        r = [100 * (1 - d[p][R]["running"]["A_prime"]["E_mJ"] / d[p][R]["running"]["B4_latency_mJ"]) for R in ("30", "300", "3000")]
        b1 = [100 * (d[p][R]["gated"]["B1_uniform"]["E_mJ"] / d[p][R]["gated"]["A_prime"]["E_mJ"] - 1) for R in ("300", "3000")]
        print(f"w={w:<4} {p:9s} cont gap {c['gap_B4_vs_opt_pct']}% | A'run vs B4run {[round(x,1) for x in r]} | B5 gap {[round(x,2) for x in g]} | B1 gated +{[round(x,1) for x in b1]}")
