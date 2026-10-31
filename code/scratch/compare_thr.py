"""Margin table before (tables_corr) and after (tables_thr) the threshold-consistent costs."""
import json
import sys

S = sys.argv[1]
for V in ("lo", "hi"):
    old = json.load(open(f"{S}/cost/threshold_sweep_corr_{V}.json"))
    new = json.load(open(f"{S}/cost/threshold_sweep_thr_{V}.json"))
    for pset in ("nominal", "z2_anchor"):
        print(f"== {V} / {pset}: margin | A' bits old -> new | B1 bits, gap old -> new | B2 | B3 | B4 | jit300 B1 | jit3000 B1")
        for m in old:
            o, n = old[m][pset], new[m][pset]
            g = lambda r, k, x="cont": r[x].get(k)  # noqa: E731
            print(f"  {m:>4} | {o['cont_bits']['A_prime']} -> {n['cont_bits']['A_prime']} | "
                  f"{o['cont_bits']['B1_uniform']} {g(o, 'B1_uniform'):+.3f} -> {n['cont_bits']['B1_uniform']} {g(n, 'B1_uniform'):+.3f} | "
                  f"{g(o, 'B2_proxy'):+.3f} -> {g(n, 'B2_proxy'):+.3f} | {g(o, 'B3_sequential'):+.3f} -> {g(n, 'B3_sequential'):+.3f} | "
                  f"{g(o, 'B4_latency'):+.3f} -> {g(n, 'B4_latency'):+.3f} | "
                  f"{g(o, 'B1_uniform', 'jit_300'):+.3f} -> {g(n, 'B1_uniform', 'jit_300'):+.3f} | "
                  f"{g(o, 'B1_uniform', 'jit_3000'):+.3f} -> {g(n, 'B1_uniform', 'jit_3000'):+.3f}")
