"""Compact digest of the accuracy-constrained outputs (search, pareto, proxy, idle)."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
C = sys.argv[1]  # cost dir


def load(name):
    p = os.path.join(C, name)
    return json.load(open(p)) if os.path.exists(p) else None


a = load("acc_model.json")
if a:
    print("threshold", a.get("threshold"), "| keys", [k for k in a if k != "acc"])
s = load("search_acc.json")
if s:
    for pname in ("nominal", "z2_anchor", "bismo_fit", "flat"):
        r = s[pname]
        print(f"== search {pname}")
        for reg, res in [("cont", r["continuous"])] + [("R=" + k, v) for k, v in r["fixed_rate"].items()]:
            ref = res.get("B4_latency", {})
            refE = ref.get("E_fixed_mJ", ref.get("E_continuous_mJ")) if reg != "cont" else ref.get("E_continuous_mJ")
            parts = []
            for k in ("A_prime", "B1_uniform", "B2_proxy", "B3_sequential", "B4_latency"):
                d = res.get(k)
                if not d:
                    continue
                e = d.get("E_fixed_mJ", d["E_continuous_mJ"]) if reg != "cont" else d["E_continuous_mJ"]
                parts.append(f"{k.split('_')[0]}={e} ({'' if not refE or e is None else f'{100*(e/refE-1):+.1f}%'}) "
                             f"{''.join(map(str, d['bits']))}/{d['mode'][:3]}/{d['f_MHz']:.0f}/acc{d['acc_est']}")
            print(f"  {reg:7s} " + " | ".join(parts))
p = load("pareto.json")
if p:
    for pname in ("nominal", "z2_anchor"):
        print(f"== pareto {pname} (A' vs B1 saving % by accuracy floor)")
        for reg, tab in p[pname].items():
            print(f"  {reg:12s} " + " ".join(f"{t['acc_floor']:.0f}:{t['saving_pct']}" for t in tab))
pr = load("proxy_acc.json")
if pr:
    print("== proxy_acc keys", list(pr)[:10])
