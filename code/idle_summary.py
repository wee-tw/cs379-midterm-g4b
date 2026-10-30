"""Summarise energy_search.py idle output: gaps between idle policies per power set (model)."""
import json, sys
d = json.load(open(sys.argv[1]))
for p, res in d.items():
    c = res["continuous"]
    s_run, g_b4, g_vs_run, low_vs_gate = [], [], [], []
    for R, r in res.items():
        if R == "continuous":
            continue
        run, gate, low = r["running"], r["gated"], r["lowclk"]
        s_run.append(100 * (1 - run["A_prime"]["E_mJ"] / run["B4_latency_mJ"]))
        g_b4.append(100 * (gate["B4_latency_mJ"] / gate["A_prime"]["E_mJ"] - 1))
        g_vs_run.append(100 * (1 - gate["B4_latency_mJ"] / run["A_prime"]["E_mJ"]))
        low_vs_gate.append(100 * (low["A_prime"]["E_mJ"] / gate["A_prime"]["E_mJ"] - 1))
    f = lambda v: f"{min(v):.2f}..{max(v):.2f}"
    print(f"{p:10s} cont gap {c['gap_B4_vs_opt_pct']}% (bound {c['bound_pct']}%) | A'run vs B4run saving {f(s_run)} | "
          f"B4+halt above gated opt {f(g_b4)} | B4+halt saving vs A'run {f(g_vs_run)} | lowclk above gated {f(low_vs_gate)}")
    print("   per R (30,300,1000,3000,5000): A'run-vs-B4", [round(x, 1) for x in s_run],
          "| B4halt-gap", [round(x, 2) for x in g_b4], "| B4halt-vs-A'run", [round(x, 1) for x in g_vs_run],
          "| lowclk-gap", [round(x, 2) for x in low_vs_gate])
