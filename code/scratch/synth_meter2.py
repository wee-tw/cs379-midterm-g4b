"""Synthetic meter log for a mock session (test only): latency-aware model with a pipeline
fill of L = 4.6 II cycles, so batched rate_halt windows have a known a + b / k."""
import csv, json, sys
import numpy as np
d = sys.argv[1]
P0, kf, kfu, kact = 1.30, 0.0030, 0.0010, 0.0015
U = {"cnv-w1a1": 0.64, "cnv-w1a2": 0.70, "cnv-w2a2": 0.55}
cyc, L = 16384, 4.6 * 16384
rows = list(csv.DictReader(open(d + "/windows.csv")))
rng = np.random.default_rng(1)
out = []
for r in rows:
    if not r.get("pc_t0") or r.get("ok") not in ("True", "true"):
        continue
    u, f = U[r["model"]], float(r["fclk0_mhz"])
    st, ips = r["state"], float(r["images_per_s"] or 0)
    run = float(r["run_mhz"] or f)
    k = int(float(r.get("k") or 1))
    clk = kf + kfu * u
    if st == "idle": P = P0 + clk * f
    elif st == "idle_halt": P = P0
    elif st == "idle_slow": P = P0 + clk * 0.252
    elif st == "cont": P = P0 + (clk + kact * u) * f
    elif st == "single": P = P0 + clk * f + kact * u * f * cyc / L
    elif st in ("rate", "jit"): P = P0 + clk * f + kact * u * f * min(1.0, ips * cyc / (f * 1e6))
    elif st == "rate_halt":
        on = min(1.0, ips * (L + (k - 1) * cyc) / k / (run * 1e6))
        P = P0 + clk * run * on + kact * u * run * min(1.0, ips * cyc / (run * 1e6))
    elif st == "rate_slow":
        on = min(1.0, ips * L / (run * 1e6))
        P = P0 + clk * run * on + kact * u * ips * cyc / 1e6 + clk * 0.252 * (1 - on)
    else:
        continue
    t0, t1 = float(r["pc_t0"]), float(r["pc_t1"])
    for t in np.arange(t0, t1, 0.05):
        out.append((t, P + rng.normal(0, 0.002)))
with open(d + "/meter_synth.csv", "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["t", "p"]); w.writerows(out)
json.dump(U, open(d + "/util_synth.json", "w"))
print(len(out), "samples; expected fill cycles", L - cyc)
