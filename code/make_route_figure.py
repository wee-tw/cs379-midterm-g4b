"""Figure: post-route against post-synthesis period of the routed critical nodes (notes 12.41).

  python make_route_figure.py --csv ../results/audit_dataset/timing_route.csv \
      --out ../results/figs/timing_route_check.png
"""

import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_timing_figure import MUTED, SERIES, TEXT


def kind(node):
    if node.startswith("RTL"):
        return "RTL MVU"
    if node.startswith("HLS"):
        return "HLS MVU, SF = 1" if node.endswith(" SF1") else "HLS MVU, SF > 1"
    return "sliding window"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    lo, hi = 3.5, 14.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(9.2, 8.35, "dotted: post-route = post-synthesis", fontsize=7, color=MUTED, ha="left")
    for ns in range(8, 15):   # clock class boundaries: FCLK0 = 1000/ns MHz
        ax.axhline(ns, color=MUTED, lw=0.8 if ns <= 10 else 0.5, ls=(0, (4, 3)), zorder=1)
        ax.text(hi - 0.05, ns + 0.05, f"{1000 / ns:.0f} MHz", fontsize=6.5, color=MUTED,
                va="bottom", ha="right")
    groups = ["RTL MVU", "HLS MVU, SF > 1", "HLS MVU, SF = 1", "sliding window"]
    markers = ["o", "^", "D", "s"]
    for g, color, marker in zip(groups, SERIES, markers):
        pts = [r for r in rows if kind(r["node"]) == g]
        ax.scatter([float(r["post_synth_ns"]) for r in pts], [float(r["post_route_ns"]) for r in pts],
                   s=40, marker=marker, color=color, edgecolors="white", linewidths=0.8, zorder=3,
                   label=f"{g} ({len(pts)})")
    for r in rows:
        if r["verdict"] != "same":
            x, y = float(r["post_synth_ns"]), float(r["post_route_ns"])
            simd = re.search(r"SIMD(\d+)", r["node"]).group(1)
            ax.annotate(f"{r['node'][:3]} SIMD {simd}: model {float(r['model_class_MHz']):.0f} MHz, "
                        f"routed {float(r['route_class_MHz']):.0f} MHz", (x, y), xytext=(lo + 0.3, y - 0.4),
                        fontsize=6.5, color=TEXT, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Post-synthesis critical path (ns)")
    ax.set_ylabel("Post-route critical path (ns), 8 ns constraint")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    same = sum(r["verdict"] == "same" for r in rows)
    ax.set_title(f"Routed critical nodes: {same} of {len(rows)} keep their clock class "
                 "(Vivado 2022.2, OOC)", fontsize=8.5, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor="white")
    print("saved", args.out, len(rows), "nodes")


if __name__ == "__main__":
    main()
