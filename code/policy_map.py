"""H2' deepened: which PL clock policy minimises energy for a frame rate R and a per-frame
latency deadline D? (energy model and FINN latency estimates, not measurements)

A FINN dataflow design accepts a frame every II cycles, but one frame needs L cycles to
get through the pipeline (L/II = 7-23 for ResNet-8). With the PS clock throttle the clock
can be stopped between frames, and energy per frame is then set by how many clock cycles
run per frame (the per-cycle energy (k_f + k_fU U) / 1e6 J does not depend on f):
  stream   frames flow back to back at the slowest FCLK0 that keeps up with R and meets
           the deadline (L / f <= D): R II <= f and f >= L / D, cycles per frame f / R
  halt k   collect k frames, run them as one batch at fmax, stop the clock: cycles per
           frame (L + (k - 1) II) / k, worst latency (k - 1) / R + (L + (k - 1) II) / fmax
With x = D R (deadline in frame periods) and r = L / II, ignoring clock discreteness and
batch run time, streaming at the deadline clock costs L / x cycles per frame (II once
x >= r) and the largest batch, k = floor(1 + x), costs II + (L - II) / k. Even a
fractional batch (k = 1 + x) beats streaming only when x < sqrt(r); with whole batches,
batching wins in bands just above each whole number of frame periods: for k = n + 1 it
wins for n < x < r / (1 + (r - 1) / (n + 1)).

  python policy_map.py --points cost/points.npz --latency cost/latency.npz --out figs
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from energy_search import ALL_CONFIGS, F_JIT, F_SET, POWER, energies, utilisation  # noqa: E402

REGIONS = ["stream, throughput clock", "stream, deadline clock", "batched halt (k >= 2)",
           "one frame, halt", "infeasible"]
COLOURS = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#d9d9d6"]


def design(rows, lat, bits, pname):
    """Lowest continuous-energy design point of one bit configuration."""
    idx = {"".join(map(str, b)): i for i, b in enumerate(ALL_CONFIGS)}
    m = np.where(rows[:, 0].astype(int) == idx[bits])[0]
    _, e_c, _, _ = energies(rows, POWER[pname], max(F_SET))
    i = m[np.argmin(e_c[m])]
    return {"II": float(rows[i, 3]), "L": float(lat[i]),
            "U": float(utilisation(rows[i, 4], rows[i, 5], rows[i, 6]))}


def policies(d, p, R, D, fmax, kmax=64):
    """PL energy per frame (J, without P0 / R) of the best stream and best halt policy."""
    ii, L, U = d["II"], d["L"], d["U"]
    e_cyc = (p["k_f"] + p["k_fU"] * U) * 1e-6     # clock-tree energy per running cycle
    e_act = p["k_act"] * U * ii * 1e-6            # switching energy of the frame itself
    need = max(R * ii, L / D) / 1e6               # MHz
    f = F_JIT[(F_JIT >= need - 1e-9) & (F_JIT <= fmax)]
    stream = (e_cyc * f.min() * 1e6 / R + e_act, float(f.min())) if len(f) else (np.inf, None)
    halt = (np.inf, None)
    for k in range(1, kmax + 1):
        cyc = L + (k - 1) * ii
        t_run = cyc / (fmax * 1e6)
        if t_run > k / R or (k - 1) / R + t_run > D:
            break
        halt = (e_cyc * cyc / k + e_act, k)
    return stream, halt


def classify(stream, halt, R, d):
    if not np.isfinite(stream[0]) and not np.isfinite(halt[0]):
        return 4
    if stream[0] <= halt[0]:
        return 0 if stream[1] <= R * d["II"] / 1e6 * 1.0001 or stream[1] == F_JIT[
            F_JIT >= R * d["II"] / 1e6 - 1e-9].min() else 1
    return 2 if halt[1] >= 2 else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--latency", required=True)
    ap.add_argument("--bits", default="44444488")
    ap.add_argument("--lat-scale", type=float, nargs="+", default=[0.6024, 0.25, 1.0])
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = np.load(args.points)["rows"]
    lat = np.load(args.latency)["latency_cycles"]
    p, fmax = POWER[args.power], max(F_SET)
    Rs = np.logspace(1, np.log10(7000), 120)
    Ds = np.logspace(-4, 0, 120)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for scale in args.lat_scale:
        d = design(rows, lat, args.bits, args.power)
        d["L"] *= scale
        r = d["L"] / d["II"]
        grid = np.zeros((len(Ds), len(Rs)), dtype=int)
        for j, R in enumerate(Rs):
            for i, D in enumerate(Ds):
                grid[i, j] = classify(*policies(d, p, R, D, fmax), R, d)
        # how often does the square-root rule pick the same family (stream vs halt)?
        agree = tot = 0
        for j, R in enumerate(Rs):
            for i, D in enumerate(Ds):
                if grid[i, j] in (0, 1, 2, 3) and D * R >= 1 and D >= d["L"] / (fmax * 1e6) * 2:
                    tot += 1
                    agree += (grid[i, j] in (0, 1)) == (D * R > np.sqrt(r))
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.pcolormesh(Rs, Ds * 1e3, grid, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=4.5,
                      shading="nearest", rasterized=True)
        for c, lab, ls in ((r, "D R = L/II", "-"), (np.sqrt(r), "D R = sqrt(L/II)", "--"),
                           (1.0, "D R = 1", ":")):
            ax.plot(Rs, c / Rs * 1e3, c="#222222", lw=1, ls=ls, label=lab)
        ax.axhline(d["L"] / (fmax * 1e6) * 1e3, c="#222222", lw=0.8, ls="-.",
                   label="D = L / fmax")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(Ds[0] * 1e3, Ds[-1] * 1e3)
        ax.set_xlabel("frame rate R (frames/s)", fontsize=8)
        ax.set_ylabel("latency deadline D (ms)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Lowest-energy clock policy, {args.bits}, L/II = {r:.1f} "
                     f"(model, '{args.power}')", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOURS]
        leg1 = ax.legend(handles, REGIONS, fontsize=6, loc="upper right", frameon=True)
        ax.add_artist(leg1)
        ax.legend(fontsize=6, loc="lower left", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"policy_map_{args.power}_L{scale:g}.png"), dpi=200)
        plt.close(fig)
        # slices: clock cycles per frame (in units of II) against the deadline in frame
        # periods; this does not depend on the power parameters
        xs = np.linspace(0.5, 1.5 * r, 600)
        e_cyc = (p["k_f"] + p["k_fU"] * d["U"]) * 1e-6
        e_act = p["k_act"] * d["U"] * d["II"] * 1e-6
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        for R, ls in ((30.0, "-"), (300.0, "--")):
            cs, ch = [], []
            for x in xs:
                (es, _), (eh, _) = policies(d, p, R, x / R, fmax)
                cs.append((es - e_act) / e_cyc / d["II"] if np.isfinite(es) else np.nan)
                ch.append((eh - e_act) / e_cyc / d["II"] if np.isfinite(eh) else np.nan)
            ax.plot(xs, cs, c=COLOURS[1], lw=1.5, ls=ls, label=f"stream, {R:.0f} FPS")
            ax.plot(xs, ch, c=COLOURS[2], lw=1.5, ls=ls, label=f"batched halt, {R:.0f} FPS")
        ax.axvline(np.sqrt(r), c="#555555", lw=0.8, ls=":")
        ax.annotate("sqrt(L/II)", (np.sqrt(r), r * 0.9), fontsize=6, color="#555555",
                    xytext=(3, 0), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel("deadline D x rate R (frame periods)", fontsize=8)
        ax.set_ylabel("clock cycles per frame / II", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, frameon=False)
        ax.set_title(f"Cycles per frame of the best policy in each family, L/II = {r:.1f} "
                     "(model)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"policy_slices_L{scale:g}.png"), dpi=200)
        plt.close(fig)
        R = 300.0
        sl = [policies(d, p, R, D, fmax) for D in Ds]
        summary[str(scale)] = {
            "design": d, "L_over_II": round(r, 2), "sqrt_rule": round(float(np.sqrt(r)), 2),
            "sqrt_rule_agreement": round(agree / max(tot, 1), 3), "cells_checked": tot,
            "slice_300fps": [{"D_ms": round(D * 1e3, 3),
                              "stream_mJ": None if not np.isfinite(s[0]) else round(s[0] * 1e3, 4),
                              "stream_MHz": s[1],
                              "halt_mJ": None if not np.isfinite(h[0]) else round(h[0] * 1e3, 4),
                              "halt_k": h[1]} for D, (s, h) in zip(Ds[::8], sl[::8])]}
        print(f"L/II {r:5.1f} (lat-scale {scale}): sqrt rule {np.sqrt(r):.2f} frame periods; "
              f"agrees with the exact policy choice in {100 * agree / max(tot, 1):.1f}% of "
              f"{tot} cells")
    json.dump(summary, open(os.path.join(args.out, f"policy_map_{args.power}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
