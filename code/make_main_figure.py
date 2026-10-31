"""Figure: main result with measured accuracy and timing classes (implementation_notes.md 12.39-12.40).

Energy per frame of B1 and B2 above the verified A' at each accuracy margin, one panel per
routing margin on the post-synthesis periods (timing_safe.py table outputs). Baselines are
re-picked with measured accuracy (B1_measured, B2_measured); a bar whose re-pick is not
available falls back to the model's pick and is hatched.

  python make_main_figure.py --tables 0=main_table_m0.json 10=main_table_m0.1.json \
      20=main_table_m0.2.json --power nominal --out ../results/figs/main_result_timing.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_timing_figure import MUTED, SERIES, TEXT


def value(row, measured, model):
    """(relative energy, picked by the model?) of a baseline."""
    d = row.get(measured)
    if d and d.get("bits") and d.get("rel_to_A") is not None:
        return d["rel_to_A"], False
    return row[model]["rel_to_A"], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True,
                    help="<routing margin %%>=<table json>, or <panel title>=<table json>")
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="extra footnote line")
    args = ap.parse_args()
    panels = [(k, json.load(open(v))) for k, v in (s.split("=", 1) for s in args.tables)]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "hatch.color": "white"})
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels) + 0.4, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    top, hatched = 0.0, False
    for ax, (rm, t) in zip(axes, panels):
        margins = sorted(t, key=float)
        x = np.arange(len(margins))
        for j, (name, meas, model) in enumerate([("B1 uniform", "B1_measured", "B1_uniform"),
                                                 ("B2 BOPs + FINN default", "B2_measured", "B2_proxy")]):
            vals = [value(t[m][args.power], meas, model) for m in margins]
            y = np.array([100 * v for v, _ in vals])
            top = max(top, y.max())
            bars = ax.bar(x + (j - 0.5) * 0.38, y, width=0.36, color=SERIES[j], edgecolor="white",
                          linewidth=0.8, label=name, zorder=3)
            for b, (v, fallback), yy in zip(bars, vals, y):
                if fallback:
                    hatched = True
                    b.set_hatch("////")
                label = int(abs(yy) + 0.5) * (1 if yy >= 0 else -1)   # half up, as in the text
                ax.text(b.get_x() + b.get_width() / 2, max(yy, 0) + 1.2, f"{label}", ha="center",
                        va="bottom", fontsize=6.5, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        labels = []
        for m in margins:
            a = t[m][args.power]["A_verified"]
            labels.append(f"{float(m):g}\n{a['E_mJ']:.3f} mJ")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_xlabel("accuracy margin (points)\nverified A′ energy per frame", fontsize=7)
        title = rm if not rm.replace(".", "").isdigit() else (
            f"post-synthesis periods +{rm}%" if rm != "0" else "post-synthesis periods")
        ax.set_title(title,
                     fontsize=8, color=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("energy per frame above A′ (%)")
    axes[0].set_ylim(min(0, axes[0].get_ylim()[0]), top * 1.15)
    handles, names = axes[0].get_legend_handles_labels()
    if hatched:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch="////", edgecolor="white"))
        names.append("model's pick (no measured re-pick)")
    fig.legend(handles, names, loc="upper center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, f"Energy is a model estimate ('{args.power}' power parameters); accuracy "
             "of A′ and of the re-picked baselines is measured (3–5 seeds per configuration)." + (
                 "\n" + args.note if args.note else ""), ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
