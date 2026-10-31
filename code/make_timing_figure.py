"""Figure: post-synthesis critical path of FINN v0.10.1 MVUs against SIMD (notes 12.35-12.36).

  python make_timing_figure.py --hls ../results/audit_dataset/timing_hls.csv \
      --rtl ../results/audit_dataset/timing_rtl.csv --out ../results/figs/timing_vs_simd.png
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette (dataviz skill, slots 1-4 in order) and text tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED = "#0b0b0b", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hls", required=True)
    ap.add_argument("--rtl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hls = list(csv.DictReader(open(args.hls, encoding="utf-8")))
    rtl = [r for r in csv.DictReader(open(args.rtl, encoding="utf-8")) if r["op"] == "MVAU_rtl"]
    groups = [
        ("RTL MVU, 4-bit core", "o",
         [(int(r["SIMD"]), float(r["period_ns_clock_in_synthesis"])) for r in rtl
          if int(r["weight_bits"]) <= 4 and int(r["input_bits"]) <= 4]),
        ("RTL MVU, 8-bit core", "s",
         [(int(r["SIMD"]), float(r["period_ns_clock_in_synthesis"])) for r in rtl
          if not (int(r["weight_bits"]) <= 4 and int(r["input_bits"]) <= 4)]),
        ("HLS MVU, input folded (SF > 1)", "^",
         [(int(r["SIMD"]), float(r["period_ns"])) for r in hls if int(r["SF"]) > 1]),
        ("HLS MVU, not folded (SF = 1)", "D",
         [(int(r["SIMD"]), float(r["period_ns"])) for r in hls if int(r["SF"]) == 1]),
    ]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for ns, mhz in ((8.0, 125), (9.0, 111), (10.0, 100)):
        ax.axhline(ns, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(330, ns, f"{mhz} MHz", va="center", ha="left", fontsize=8, color=MUTED)
    for (label, marker, pts), color in zip(groups, SERIES):
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, marker=marker, color=color,
                   edgecolors="white", linewidths=0.8, label=f"{label} ({len(pts)})", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.set_xlim(0.8, 300)
    ax.set_xlabel("SIMD (input lanes per PE)")
    ax.set_ylabel("Post-synthesis critical path (ns)")
    # no grid: the three clock lines are the reference
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_title("FINN v0.10.1 MVUs on xc7z020 (Vivado / Vitis HLS 2022.2, out of context)",
                 fontsize=9, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor="white")
    print("saved", args.out, {g[0]: len(g[2]) for g in groups})


if __name__ == "__main__":
    main()
