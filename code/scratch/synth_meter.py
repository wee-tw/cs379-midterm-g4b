"""Synthetic meter log for a mock session, from a known power model (test only)."""
import csv, json, sys
import numpy as np
d = sys.argv[1]
P0, kf, kfu, kact = 1.30, 0.0030, 0.0010, 0.0015
U = {"cnv-w1a1": 0.64, "cnv-w1a2": 0.70, "cnv-w2a2": 0.55}
cyc = 16384
rows = list(csv.DictReader(open(d + "/windows.csv")))
rng = np.random.default_rng(1)
out = []
for r in rows:
    u, f = U[r["model"]], float(r["fclk0_mhz"])
    st, ips = r["state"], float(r["images_per_s"] or 0)
    run = float(r["run_mhz"] or f)
    if st == "idle": P = P0 + (kf + kfu * u) * f
    elif st == "idle_halt": P = P0
    elif st == "idle_slow": P = P0 + (kf + kfu * u) * 0.252
    elif st == "cont": P = P0 + (kf + kfu * u + kact * u) * f
    elif st == "rate":
        duty = min(1.0, ips * cyc / (f * 1e6)); P = P0 + (kf + kfu * u) * f + kact * u * f * duty
    elif st == "rate_halt":
        duty = min(1.0, ips * cyc / (run * 1e6)); P = P0 + (kf + kfu * u + kact * u) * run * duty
    elif st == "rate_slow":
        duty = min(1.0, ips * cyc / (run * 1e6))
        P = P0 + (kf + kfu * u + kact * u) * run * duty + (kf + kfu * u) * 0.252 * (1 - duty)
    t0, t1 = float(r["pc_t0"]), float(r["pc_t1"])
    for t in np.arange(t0, t1, 0.05):
        out.append((t, P + rng.normal(0, 0.005)))
with open(d + "/meter_synth.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t", "p"]); w.writerows(out)
json.dump(U, open(d + "/util_synth.json", "w"))
print(len(out), "samples")
