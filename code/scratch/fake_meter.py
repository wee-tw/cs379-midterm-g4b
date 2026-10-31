"""Synthetic USB-meter log for a mock session (pipeline test only): fnirsi_logger-style text
(timestamp sample_in_packet voltage_V current_A) at 10 Hz, power from a simple model."""
import csv, json, sys
import numpy as np
sdir, out = sys.argv[1], sys.argv[2]
util = json.load(open(sys.argv[3]))
rows = list(csv.DictReader(open(f"{sdir}/windows.csv")))
P0, kf, kfU, kact = 1.60, 0.0020, 0.0030, 0.0040
def power(r):
    f = float(r["fclk0_mhz"] or 100); U = util.get(r["model"], 0.5); s = r["state"]
    idle = P0 + (kf + kfU * U) * f
    if s == "idle_halt": return P0
    if s == "idle_slow": return P0 + (kf + kfU * U) * 0.25
    if s in ("cont", "jit"): return idle + kact * U * f
    if s == "single": return idle + 0.5 * kact * U * f
    return idle + 0.2 * kact * U * f
t0 = float(rows[0]["pc_t0"]) - 5; t1 = float(rows[-1]["pc_t1"]) + 5
ts = np.arange(t0, t1, 0.1)
pw = np.full(ts.shape, P0 + 0.3)
for r in rows:
    a, b = float(r["pc_t0"]), float(r["pc_t1"])
    pw[(ts >= a) & (ts <= b)] = power(r)
rng = np.random.default_rng(0)
with open(out, "w") as f:
    f.write("timestamp sample_in_packet voltage_V current_A\n")
    for t, p in zip(ts, pw + rng.normal(0, 0.01, ts.shape)):
        f.write(f"{t:.3f} 0 5.0000 {p / 5.0:.5f}\n")
print("samples", len(ts))
