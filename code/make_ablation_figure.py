"""Figure: energy of the verified A' with and without the timing-safe refold (notes 12.44).

Same clock-class model and the same measured configurations for every variant; the label on
each bar is the clock of the design.

  python make_ablation_figure.py --tables "no refold=main_table_calv2c_r.json" \
      "refold, SIMD ≤ 27=main_table_tsr_v2c_r.json" \
      "refold, RTL ≤ 12, HLS ≤ 18=main_table_tsr_v1218_m0.json" --out ../results/figs/refold_ablation.png

Labels must not contain "=" (the label ends at the first "="), so write ≤, not <=.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_timing_figure import MUTED, SERIES, TEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True, help="<label>=<timing_safe.py table json>")
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    variants = [(k, json.load(open(v))) for k, v in (s.split("=", 1) for s in args.tables)]
    margins = sorted(variants[0][1], key=float)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(margins))
    w = 0.8 / len(variants)
    for j, (label, t) in enumerate(variants):
        rows = [t[m][args.power]["A_verified"] for m in margins]
        bars = ax.bar(x + (j - (len(variants) - 1) / 2) * w, [r["E_mJ"] for r in rows], width=w * 0.92,
                      color=SERIES[j], edgecolor="white", linewidth=0.8, label=label, zorder=3)
        for b, r in zip(bars, rows):
            ax.text(b.get_x() + b.get_width() / 2, r["E_mJ"] + 0.004, f"{r['f_MHz']:.0f}", ha="center",
                    va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(m):g}" for m in margins])
    ax.set_xlabel("accuracy margin (points)")
    ax.set_ylabel("verified A′ energy per frame (mJ)")
    ax.set_ylim(0, max(t[m][args.power]["A_verified"]["E_mJ"] for _, t in variants for m in margins) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=len(variants), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.text(0.5, -0.03, f"Numbers above the bars: clock (MHz). Energy is a model estimate ('{args.power}'); "
             "the accuracy of every A′ is measured (3–5 seeds).", ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
