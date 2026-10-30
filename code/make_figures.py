"""Figures for the midterm paper/slides. Each figure is skipped if its inputs are missing.

  python make_figures.py --runs exp3 --recipe exp2 --points points.npz \
      --acc acc_model.json --out ../results/figs

Labels follow the plan's rule: every plot says whether its numbers are measured
(accuracy, QAT on the CIFAR-10 test set) or estimated (FINN v0.10.1 estimates and the
literature-calibrated power model).
"""

import argparse
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from energy_search import ALL_CONFIGS, F_SET, POWER, bops, energies, is_uniform  # noqa: E402

PAIRS = list(itertools.product((2, 4, 8), repeat=2))


def last3(path):
    """Final-epoch accuracy (OneCycle ends at a near-zero learning rate)."""
    r = json.load(open(path))
    if "history" not in r:
        return None
    return r["history"][-1]["test_acc"]


def fig_uniform(runs, out):
    vals = {}
    for w, a in PAIRS:
        f = os.path.join(runs, "sens_base0.json" if (w, a) == (4, 4) else f"b1_w{w}a{a}.json")
        if os.path.exists(f):
            vals[(w, a)] = last3(f)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = [f"W{w}A{a}" for (w, a) in vals]
    ax.bar(labels, list(vals.values()), color="#4C72B0")
    ax.axhline(86.25, ls="--", c="k", lw=1, label="float (MLPerf ResNet-8) 86.25%")
    ax.axhline(86.01, ls=":", c="gray", lw=1, label="float (FINN-style) 86.01%")
    ax.set_ylabel("CIFAR-10 top-1 (%)")
    ax.set_ylim(min(vals.values()) - 5, 90)
    ax.set_title("Uniform precision, 30-epoch QAT (measured)")
    ax.legend(fontsize=7, loc="lower right")
    for i, v in enumerate(vals.values()):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "uniform_accuracy.png"), dpi=200)


def fig_sensitivity(runs, out):
    base_f = os.path.join(runs, "sens_base.json")
    if not os.path.exists(base_f):
        return
    base = last3(base_f)
    M = np.full((4, 9), np.nan)
    for g in range(4):
        for j, (w, a) in enumerate(PAIRS):
            if (w, a) == (4, 4):
                M[g, j] = 0.0
                continue
            f = os.path.join(runs, f"sens_g{g + 1}_w{w}a{a}.json")
            if os.path.exists(f):
                M[g, j] = last3(f) - base
    fig, ax = plt.subplots(figsize=(7, 2.6))
    im = ax.imshow(M, cmap="RdBu", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"W{w}A{a}" for w, a in PAIRS], fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["G1 stem+S1", "G2 S2", "G3 S3", "G4 FC"], fontsize=7)
    for g in range(4):
        for j in range(9):
            if not np.isnan(M[g, j]):
                ax.text(j, g, f"{M[g, j]:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ top-1 vs all-W4A4 (pts)")
    ax.set_title(f"Per-block sensitivity, base {base:.2f}% (measured, 8-epoch fine-tunes)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sensitivity.png"), dpi=200)


def fig_hw(rows, out):
    """Left: LUT-only implementation (HLS MVAUs, thresholds inside); right: DSP mode."""
    cfg = rows[:, 0].astype(int)
    mode = rows[:, 1].astype(int)
    fps = 100e6 / rows[:, 3]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (mi, col, ylab, cap, title) in zip(axes, [
            (0, 4, "LUT (FINN estimate)", 53200, "LUT-only (HLS MVAU) implementation"),
            (1, 6, "DSP (FINN estimate)", 220, "DSP mode (RTL MVU, standalone thresholds)")]):
        for w, a in [(8, 8), (4, 4), (2, 4), (2, 2)]:
            c = ALL_CONFIGS.index((w, a) * 4)
            m = (cfg == c) & (mode == mi)
            if not m.any():
                continue
            order = np.argsort(fps[m])
            x, y = fps[m][order], rows[m, col][order]
            best = np.minimum.accumulate(y[::-1])[::-1]  # min resource reaching >= x
            ax.step(x, best, where="post", label=f"uniform W{w}A{a}")
        ax.axhline(cap, c="k", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("throughput at 100 MHz (FPS)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)
    axes[0].axhline(53200 * 0.8, c="gray", lw=1, ls="--")
    axes[0].text(110, 43500, "80% budget", fontsize=6)
    fig.suptitle("Resources needed for a throughput, feasible points only "
                 "(estimated, FINN v0.10.1; RTL-MVU LUTs are not estimated)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lut_vs_throughput.png"), dpi=200)


def fig_energy(rows, acc, out, pname="nominal"):
    p = POWER[pname]
    cfg = rows[:, 0].astype(int)
    n = len(ALL_CONFIGS)
    uni = np.array([is_uniform(b) for b in ALL_CONFIGS])
    a = np.array([acc.get(",".join(map(str, b)), np.nan) for b in ALL_CONFIGS])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, (title, R) in zip(axes, [("continuous", None), ("fixed 300 FPS", 300.0),
                                     ("fixed 3000 FPS", 3000.0)]):
        e = np.full(n, np.inf)
        for f in F_SET:
            fps, ec, p_idle, e_dyn = energies(rows, p, f)
            v = ec if R is None else np.where(fps >= R, p_idle / R + e_dyn, np.inf)
            np.minimum.at(e, cfg, v)
        ok = np.isfinite(e) & np.isfinite(a)
        ax.scatter(a[ok & ~uni], e[ok & ~uni] * 1e3, s=2, c="#999999", alpha=0.4,
                   label="mixed precision")
        ax.scatter(a[ok & uni], e[ok & uni] * 1e3, s=18, c="#C44E52", label="uniform")
        # Pareto fronts: lowest energy at or above each accuracy
        for mask, col, lab in [(ok, "#4C72B0", "A' front"), (ok & uni, "#C44E52", "B1 front")]:
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(-a[idx])]
            best, xs, ys = np.inf, [], []
            for i in idx:
                if e[i] < best:
                    best = e[i]
                    xs.append(a[i])
                    ys.append(best * 1e3)
            ax.step(xs, ys, where="post", c=col, lw=1.2, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("estimated top-1 (%)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("energy / inference (mJ, model)")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Energy vs accuracy (estimates; power model '{pname}')", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"energy_vs_accuracy_{pname}.png"), dpi=200)


def fig_idle(idle_json, out):
    """Fixed-rate energy per idle policy, relative to B4 with the clock running (H2')."""
    d = json.load(open(idle_json))
    tag = os.path.splitext(os.path.basename(idle_json))[0]
    fig, axes = plt.subplots(1, len(d), figsize=(2.6 * len(d), 3.0), sharey=True)
    lines = [("running", "A_prime", "#4C72B0", "-", "A', clock running (>= 50 MHz)"),
             ("running", "B1_uniform", "#4C72B0", "--", "B1, clock running"),
             ("lowclk", "A_prime", "#DD8452", "-", "A', clocks down to 5 MHz"),
             ("gated", "A_prime", "#55A868", "-", "A', idle clock stopped"),
             ("gated", "B1_uniform", "#55A868", "--", "B1, idle clock stopped")]
    for ax, (pname, res) in zip(np.atleast_1d(axes), d.items()):
        rates = sorted(int(r) for r in res if r != "continuous")
        ref = np.array([res[str(r)]["running"]["B4_latency_mJ"] for r in rates])
        for pol, who, col, ls, lab in lines:
            y = np.array([res[str(r)][pol][who]["E_mJ"] for r in rates])
            ax.plot(rates, y / ref, c=col, ls=ls, marker="o", ms=3, lw=1.2, label=lab)
        b5 = np.array([res[str(r)]["gated"]["B4_latency_mJ"] for r in rates])
        ax.plot(rates, b5 / ref, c="#C44E52", ls=":", marker="s", ms=3, lw=1.2,
                label="B5 = B4 + race-to-halt")
        ax.axhline(1.0, c="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("required rate R (FPS)")
        ax.set_title(pname, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("energy / inference ÷ B4 (clock running)")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.suptitle("Fixed-rate energy by idle policy (model estimates, not measured)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"idle_policies_{tag}.png"), dpi=200)


def fig_sweep(sweep_json, out, pname="nominal"):
    """Baselines relative to A' as the accuracy threshold loosens (threshold_sweep.py)."""
    d = json.load(open(sweep_json))
    margins = sorted(d, key=float)
    thr = [d[m]["threshold"] for m in margins]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)
    for ax, (key, title) in zip(axes, [("cont", "continuous"),
                                       ("gated_3000", "fixed 3000 FPS, idle clock stopped")]):
        for name, col, lab in [("B1_uniform", "#C44E52", "B1 uniform"),
                               ("B2_proxy", "#8172B2", "B2 BOPs + FINN default"),
                               ("B3_sequential", "#DD8452", "B3 BOPs bits, then folding"),
                               ("B4_latency", "#4C72B0", "B4 latency-opt (B5 when gated)")]:
            y = [100 * d[m][pname][key].get(name, np.nan) for m in margins]
            ax.plot(thr, y, c=col, marker="o", ms=3, lw=1.2, label=lab)
        ax.axhline(0, c="k", lw=0.6)
        ax.invert_xaxis()
        ax.set_xlabel("accuracy threshold (%, additive model)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("energy above A' (%)")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Joint vs sequential vs uniform selection (model estimates, '{pname}')",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"threshold_sweep_{pname}.png"), dpi=200)


def fig_budget(budget_json, out, pname="nominal"):
    """Budget cliff: best uniform (B1) vs A' in continuous mode as the LUT budget changes."""
    d = json.load(open(budget_json))
    b = d["budgets"]
    x = [100 * r["lut"] for r in b]
    # two panels on one x axis rather than a second y axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 3.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.plot(x, [r[pname]["B1_over_A_pct"] for r in b], c="#C44E52", marker="o", ms=3)
    ax1.set_ylabel("B1 energy\nabove A' (%)", fontsize=8)
    ax1.grid(alpha=0.25, lw=0.5)
    ax2.step(x, [r[pname]["A_prime_fps"] for r in b], where="mid", c="#4C72B0", lw=1.5, label="A'")
    ax2.step(x, [r[pname]["B1_fps"] for r in b], where="mid", c="#C44E52", lw=1.5, ls="--",
             label="B1 (best uniform)")
    ax2.set_ylabel("FPS at 125 MHz", fontsize=8)
    ax2.set_xlabel("LUT budget (% of xc7z020, BRAM budget 10 points lower)", fontsize=8)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
    ax1.set_title(f"Budget cliff, continuous mode (model estimates, '{pname}', "
                  f"accuracy >= {d['threshold']:.2f}%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"budget_cliff_{pname}.png"), dpi=200)


def fig_candidates(rows, cand_dir, roles, out, pname="nominal"):
    """Measured QAT accuracy (30 epochs) vs estimated continuous energy of each trained
    candidate configuration; roles: {bits "4,4,...": label}."""
    p = POWER[pname]
    fmax = max(F_SET)
    cfg = rows[:, 0].astype(int)
    fps, e_c, _, _ = energies(rows, p, fmax)
    best = np.full(len(ALL_CONFIGS), np.inf)
    np.minimum.at(best, cfg, e_c)
    index = {",".join(map(str, b)): i for i, b in enumerate(ALL_CONFIGS)}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for bits, label in roles.items():
        f = os.path.join(cand_dir, "cand_" + bits.replace(",", "_") + ".json")
        if not os.path.exists(f):
            continue
        acc = json.load(open(f))["history"][-1]["test_acc"]
        e = best[index[bits]] * 1e3
        col = ("#4C72B0" if label.startswith("A'") else "#C44E52" if label.startswith("B1")
               else "#8172B2")
        ax.scatter(acc, e, s=28, c=col)
        ax.annotate(f"{label}\n{bits.replace(',', '')}", (acc, e), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("measured top-1 after 30-epoch QAT (%)")
    ax.set_ylabel(f"energy / inference (mJ, model '{pname}')")
    ax.set_title("Trained candidates: measured accuracy vs estimated energy "
                 "(continuous, LUT budget 80%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"candidates_{pname}.png"), dpi=200)


def fig_recipe(recipe_dirs, out):
    rows = []
    for d in recipe_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*w*a*_*.json"))):
            name = os.path.basename(f)[:-5]
            if not (name.startswith("rs_") or name.startswith("abl_")):
                continue
            v = last3(f)
            if v is not None:
                rows.append((name.split("_", 1)[1], v))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#55A868")
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7)
    ax.set_xlabel("CIFAR-10 top-1 (%), 15-epoch QAT from float (measured)")
    ax.set_title("Quantizer recipe ablation (G4 kept at W8A8)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recipe_ablation.png"), dpi=200)


def fig_clock_timeline(out, L_over_II=5.0, frames=3):
    """Schematic (not data; illustrative II = 10% of the period at f_max): PL clock
    activity over a few frame periods under each policy. Bar height = clock frequency,
    filled area = clock cycles spent per frame."""
    P, II = 1.0, 0.1                     # frame period and initiation interval at f_max
    L = L_over_II * II                   # single-frame latency at f_max
    batch = (L + (frames - 1) * II) / frames
    rows = [(f"clock running at f_max (FINN default): {P / II:g} II per frame",
             [(0, frames * P, 1.0)], "#8172B2"),
            (f"race-to-halt, one frame at a time (B5): L = {L_over_II:g} II per frame",
             [(k * P, k * P + L, 1.0) for k in range(frames)], "#C44E52"),
            ("jit: lowest clock that sustains R, frames overlap: 1 II per frame",
             [(0, frames * P, II / P)], "#55A868"),
            (f"collect {frames} frames, then race and halt: (L + {frames - 1} II)/{frames} = "
             f"{batch / II:.2f} II per frame",
             [((frames - 1) * P, (frames - 1) * P + L + (frames - 1) * II, 1.0)], "#DD8452")]
    fig, axs = plt.subplots(len(rows), 1, figsize=(6, 3.6), sharex=True)
    for ax, (label, segs, c) in zip(axs, rows):
        for a, b, h in segs:
            ax.add_patch(plt.Rectangle((a, 0), b - a, h, color=c, alpha=0.85, lw=0))
        for k in range(frames + 1):
            ax.axvline(k * P, c="0.6", lw=0.5, ls=":")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.set_ylabel("f", fontsize=6, rotation=0, labelpad=6)
        ax.set_title(label, fontsize=7, loc="left", pad=2)
    axs[-1].set_xlim(0, frames * P)
    axs[-1].set_xticks([k * P for k in range(frames + 1)])
    axs[-1].set_xticklabels([f"{k}/R" for k in range(frames + 1)], fontsize=7)
    fig.suptitle("PL clock activity per policy (schematic): clock energy ~ area under the bar",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "clock_policies_schematic.png"), dpi=200)


def fig_latency(idle_json, out, pname="nominal"):
    """Latency-aware fixed-rate policies (energy_search.py idle --latency): energy above the
    just-in-time-clock optimum, per rate."""
    d = json.load(open(idle_json))[pname]
    Rs = sorted(int(R) for R in d if R != "continuous")
    jit = [d[str(R)]["jit"]["A_prime"]["E_mJ"] for R in Rs]
    series = [("B5: fastest design, one frame at a time, halt", "gated_lat", "B4_latency_mJ", "#C44E52", "o"),
              ("B4, clock running at 125 MHz", "running", "B4_latency_mJ", "#8172B2", "s"),
              ("A', clock running (50-125 MHz)", "running", None, "#4C72B0", "^"),
              ("A', clock running (5-125 MHz)", "lowclk", None, "#55A868", "v"),
              ("deadline 1/R: best (race-to-halt)", "dl_gated", None, "#DD8452", "D")]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for label, pol, key, c, mk in series:
        ys = []
        for R, j in zip(Rs, jit):
            v = d[str(R)][pol]["A_prime"]["E_mJ"] if key is None else d[str(R)][pol][key]
            ys.append(None if v is None else 100 * (v / j - 1))
        xs = [R for R, y in zip(Rs, ys) if y is not None]
        ax.plot(xs, [y for y in ys if y is not None], c=c, marker=mk, ms=4, lw=1, label=label,
                ls="--" if pol == "gated_lat" else "-", zorder=3 if pol == "gated_lat" else 2)
    ax.axhline(0, c="k", lw=0.6, label="jit: 44444488 at the lowest FCLK0 sustaining R (= II bound)")
    ax.text(0.99, 0.97, "B5 = deadline optimum at low R; at R >= 1000 its latency exceeds 1/R,\n"
            "so it never halts and equals B4 with the clock running", transform=ax.transAxes,
            fontsize=6, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("frame rate R (FPS)")
    ax.set_ylabel("energy per frame above jit (%)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title(f"Fixed-rate policies with single-frame latency (model estimates, '{pname}')",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"latency_policies_{pname}.png"), dpi=200)


def fig_protocol(cand_json, out, model=None):
    """Measured minus predicted top-1 (additive model) against the number of groups with
    2-bit weights, one series per training protocol (output of cand_table.py --json)."""
    d = json.load(open(cand_json))
    rows = [r for r in d["by_w2"] if model is None or r["model"] == model]
    model = model or rows[0]["model"]
    rows = [r for r in rows if r["model"] == model]
    protos = sorted({r["protocol"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 3.0))
    for i, pr in enumerate(protos):
        rs = sorted((r for r in rows if r["protocol"] == pr), key=lambda r: r["w2_groups"])
        k = np.array([r["w2_groups"] for r in rs]) + (i - (len(protos) - 1) / 2) * 0.12
        for kk, r in zip(k, rs):
            ax.scatter([kk] * len(r["errors"]), r["errors"], s=8, c=f"C{i}", alpha=0.6)
        ax.plot(k, [np.mean(r["errors"]) for r in rs], c=f"C{i}", marker="o", ms=4, lw=1,
                label=f"{pr} (mean)")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(range(5))
    ax.set_xlabel("number of groups with 2-bit weights")
    ax.set_ylabel("measured - predicted top-1 (points)")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title(f"Additive accuracy model vs QAT protocol (measured QAT, model '{model}')",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy_model_vs_protocol.png"), dpi=200)


PROTO_STYLE = {  # protocol label from cand_table.py: (legend text, colour)
    "W4A4→30 ep KD lr 0.02": ("from W4A4, 30 ep", "#2a78d6"),
    "float→30 ep KD lr 0.05": ("from float, 30 ep", "#eb6834"),
}


def fig_h3_pairs(rows, cand_json, pairs, out, pname="nominal"):
    """Each H3 pair (A' pick vs BOPs-proxy pick at one threshold) in its own panel: measured
    top-1 under each training protocol (x) against the estimated continuous energy of the
    best design for those bits (y). The energy gap is the same under both protocols; only
    the accuracy gap moves. pairs: [(title, A' bits, proxy bits)], bits as "84242448"."""
    p = POWER[pname]
    cfg = rows[:, 0].astype(int)
    _, e_c, _, _ = energies(rows, p, max(F_SET))
    best = np.full(len(ALL_CONFIGS), np.inf)
    np.minimum.at(best, cfg, e_c)
    energy = {"".join(map(str, b)): best[i] * 1e3 for i, b in enumerate(ALL_CONFIGS)}
    acc = {}
    for r in json.load(open(cand_json, encoding="utf-8"))["runs"]:
        acc[(r["config"], r["protocol"])] = r["acc"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(1.9 * len(pairs), 2.8), sharey=True)
    axes, spans = np.atleast_1d(axes), []
    for ax, (title, a_bits, b_bits) in zip(axes, pairs):
        xs, dacc = [], []
        for proto, (text, col) in PROTO_STYLE.items():
            pts = [(b, acc.get((b, proto))) for b in (a_bits, b_bits)]
            pts = [(b, v) for b, v in pts if v]
            for b, v in pts:
                sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
                ax.errorbar(np.mean(v), energy[b], xerr=sd or None,
                            fmt="o" if b == a_bits else "s", ms=6, c=col, mec="white", mew=1,
                            elinewidth=1, capsize=2, zorder=3)
                xs += [np.mean(v) - sd, np.mean(v) + sd]
            if len(pts) == 2:
                ax.plot([np.mean(v) for _, v in pts], [energy[b] for b, _ in pts], c=col,
                        lw=1, alpha=0.6, zorder=2)
                dacc.append(f"{np.mean(pts[0][1]) - np.mean(pts[1][1]):+.2f} "
                            f"{text.split(',')[0].split()[-1]}")
        spans.append((min(xs), max(xs)) if xs else None)
        gap = 100 * (energy[b_bits] / energy[a_bits] - 1)
        ax.set_title(f"{title}\nA' {a_bits} vs proxy {b_bits}\nproxy energy {gap:+.0f}%\n"
                     f"Δtop-1: {', '.join(dacc)}", fontsize=6)
    # the same x span in every panel, so a wide-looking gap is a wide gap
    width = max(hi - lo for lo, hi in filter(None, spans)) + 1.0
    for ax, s in zip(axes, spans):
        if s:
            mid = (s[0] + s[1]) / 2
            ax.set_xlim(mid - width / 2, mid + width / 2)
    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("measured top-1 (%)", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel(f"est. energy / inference (mJ, '{pname}')", fontsize=7)
    handles = [plt.Line2D([], [], c=col, marker="o", ls="-", lw=1, ms=5, label=text)
               for text, col in PROTO_STYLE.values()]
    handles += [plt.Line2D([], [], c="#555555", marker=m, ls="", ms=5, label=t)
                for m, t in (("o", "A' pick"), ("s", "BOPs-proxy pick (best folding)"))]
    handles.append(plt.Line2D([], [], ls="", label="Δtop-1 = A' minus proxy (measured)"))
    fig.legend(handles=handles, fontsize=6, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, f"h3_pairs_{pname}.png"), dpi=200)


def fig_concentration(biased_npz, truth_npz, acc_json, out, pname="nominal",
                      margins=(2, 3, 5)):
    """Where do the estimator's misjudged designs sit? Share of the top-k designs, ranked by
    the biased (FINN) energy, that the reference cost model finds infeasible, for all
    designs and within each accuracy margin (dse_harm.py; estimates)."""
    p = POWER[pname]

    def load(f):
        rows = np.load(f)["rows"]
        e = np.min(np.stack([energies(rows, p, x)[1] for x in F_SET]), axis=0)
        return rows, e

    b_rows, b_e = load(biased_npz)
    t_rows, _ = load(truth_npz)
    truth = {(int(r[0]), int(r[1]), int(r[2])) for r in t_rows}
    bad = np.array([(int(r[0]), int(r[1]), int(r[2])) not in truth for r in b_rows])
    model = json.load(open(acc_json))
    acc = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])[b_rows[:, 0].astype(int)]
    ks = np.unique(np.logspace(0, np.log10(len(bad)), 60).astype(int))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    series = [("all designs", np.arange(len(bad)))]
    series += [(f"accuracy margin {m:g}", np.where(acc >= model["reference_acc"] - m)[0])
               for m in margins]
    for (label, idx), col in zip(series, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
        o = idx[np.argsort(b_e[idx])]
        kk = ks[ks <= len(o)]
        ax.plot(kk, [100 * bad[o[:k]].mean() for k in kk], c=col, lw=1.5, label=label)
    ax.axhline(100 * bad.mean(), c="#555555", lw=0.8, ls=":")
    ax.annotate(f"overall {100 * bad.mean():.1f}%", (len(bad), 100 * bad.mean()), fontsize=6,
                color="#555555", xytext=(-4, 4), textcoords="offset points", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("top-k designs by FINN-estimated energy", fontsize=8)
    ax.set_ylabel("infeasible on the device (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("FINN-feasible designs that exceed the xc7z020 after correcting the RTL MVU "
                 "estimates", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"estimator_error_concentration_{pname}.png"), dpi=200)
    plt.close(fig)


def fig_calibration(calib_json, out, budget=42560, device=53200):
    """Design-level validation of the LUT calibration (calib_v2.py, 12.33): (a) LUT prediction error
    of the v1 calibration and of v2.1 with the design held out of the fit, against per-node
    Vivado / Vitis HLS synthesis of the selected designs; (b) the synthesized LUTs against the
    LUT budget (80% of the device). Tool-verified synthesis; the predictions are estimates."""
    rows = json.load(open(calib_json))["lodo"]
    names = {"vd:44444448_lut_14286": "A' m1, LUT impl.", "vd:44444448_dsp4c_16667": "A' m1, DSP impl.",
             "vd:44242448_dsp4c_14286": "A' m2, DSP impl.", "vd:84848484_lut_16667": "B1 m1 (v1 pick)",
             "vd2:44444448_lut_finn_16667": "B2 m1, FINN folding"}
    rows = [r for r in rows if r["design"] in names]
    rows.sort(key=lambda r: list(names).index(r["design"]))
    e1 = [100 * (r["v1_LUT"] / r["synth_LUT"] - 1) for r in rows]
    e2 = [100 * (r["lodo_LUT"] / r["synth_LUT"] - 1) for r in rows]
    col_v1, col_v2, col_s = "#a3acb5", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-5, 5, color=col_v2, alpha=0.08, lw=0)
    ax.axvline(0, c="#555555", lw=0.6)
    ax.barh(y + 0.19, e1, 0.36, color=col_v1, label="v1 calibration")
    ax.barh(y - 0.19, e2, 0.36, color=col_v2, label="v2.1, design held out")
    for yi, a, b in zip(y, e1, e2):
        for dy, v in ((0.19, a), (-0.19, b)):
            ax.text(v + (0.8 if v >= 0 else -0.8), yi + dy, f"{v:+.1f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([names[r["design"]] for r in rows], fontsize=6.5)
    ax.set_xlim(-36, 16)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("LUT prediction error vs synthesis (%)", fontsize=6.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_title("(a) design-level LUT error (shaded: ±5%)", fontsize=7.5)
    ax = axes[1]
    pct = [100 * r["synth_LUT"] / budget for r in rows]
    ax.barh(y, pct, 0.55, color=col_s)
    ax.axvline(100, c="#555555", lw=0.8, ls="--")
    ax.axvline(100 * device / budget, c="#555555", lw=0.8, ls=":")
    ax.text(99, y[-1] - 0.5, "budget", fontsize=6, color="#555555", ha="right")
    ax.text(100 * device / budget + 1, y[-1] - 0.45, "device", fontsize=6, color="#555555")
    for yi, v, r in zip(y, pct, rows):
        ax.text(v + 1.5, yi, f"{r['synth_LUT']:,.0f}", va="center", fontsize=6, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 150)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("synthesized LUTs (% of the LUT budget)", fontsize=6.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title("(b) does the design fit?", fontsize=7.5)
    fig.suptitle("Per-node Vivado / Vitis HLS 2022.2 synthesis of the selected designs (xc7z020)",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration_validation.png"), dpi=200)
    plt.close(fig)


def fig_audit(audit_json, infeasible_json, out):
    """FINN v0.10.1 estimates against out-of-context Vivado 2022.2 synthesis (xc7z020) of the
    FINN-generated RTL nodes: LUT per operator, the BRAM FINN does not estimate, and the DSPs
    of a design FINN's estimates pick (audit_rtl_ops.py vivado, prove_infeasible.py;
    tool-verified)."""
    cases = [c for c in json.load(open(audit_json)) if c["vivado"]["returncode"] == 0]

    def tot(op, src, k):
        return sum(c[src].get(k, 0) for c in cases if c["op"] == op)
    lut = [(f"RTL MVU ({sum(c['op'] == 'MVAU_rtl' for c in cases)})",
            tot("MVAU_rtl", "finn_estimate", "LUT"), tot("MVAU_rtl", "vivado", "LUT")),
           (f"RTL thresholds ({sum(c['op'] == 'Thresholding_rtl' for c in cases)})",
            tot("Thresholding_rtl", "finn_estimate", "LUT"), tot("Thresholding_rtl", "vivado", "LUT")),
           (f"SWG ({sum(c['op'] == 'ConvolutionInputGenerator_rtl' for c in cases)})",
            tot("ConvolutionInputGenerator_rtl", "finn_estimate", "LUT"),
            tot("ConvolutionInputGenerator_rtl", "vivado", "LUT")),
           (f"FMPadding ({sum(c['op'] == 'FMPadding_rtl' for c in cases)})",
            tot("FMPadding_rtl", "finn_estimate", "LUT"), tot("FMPadding_rtl", "vivado", "LUT"))]
    # the MVU's FINN BRAM is its weight memory, outside the synthesized unit; the replay buffer
    # inside it is not estimated at all
    bram = [("thresholds", 0, tot("Thresholding_rtl", "vivado", "BRAM18")),
            ("MVU replay buffer", 0, tot("MVAU_rtl", "vivado", "BRAM18"))]
    inf = json.load(open(infeasible_json))
    design = inf[0]["design"]
    dsp_finn = next(c["finn_dsp_total"] for c in inf if c.get("case") is None and c["design"] == design)
    dsp_viv = sum(c["vivado"]["DSP"] for c in inf if c.get("case") and c["design"] == design)
    col_e, col_s = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [2.4, 1.5, 1.1]})
    for ax, rows, title in ((axes[0], lut, "(a) LUT per operator"),
                            (axes[1], bram, "(b) BRAM18 not estimated")):
        y = np.arange(len(rows))[::-1]
        ax.barh(y + 0.19, [r[1] for r in rows], 0.36, color=col_e, label="FINN estimate")
        ax.barh(y - 0.19, [r[2] for r in rows], 0.36, color=col_s, label="synthesis")
        top = max(max(r[1], r[2]) for r in rows)
        for yi, r in zip(y, rows):
            for dy, v in ((0.19, r[1]), (-0.19, r[2])):
                ax.text(v + 0.01 * top, yi + dy, f"{v:,.0f}", va="center", fontsize=6, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
        ax.set_xlim(0, top * 1.25)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_title(title, fontsize=7.5)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    ax = axes[2]
    ax.bar([0, 1], [dsp_finn, dsp_viv], 0.6, color=[col_e, col_s])
    ax.axhline(220, c="#555555", lw=0.8, ls="--")
    ax.text(-0.55, 226, "device 220", fontsize=6, color="#555555", ha="left")
    for x, v in ((0, dsp_finn), (1, dsp_viv)):
        ax.text(x, v + 5, f"{v:.0f}", ha="center", fontsize=6.5, color="#333333")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FINN", "Vivado"], fontsize=6.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, dsp_viv * 1.2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title(f"(c) DSP of FINN's pick\n({design})", fontsize=7.5)
    fig.suptitle("FINN v0.10.1 estimates vs Vivado 2022.2 synthesis of the FINN-generated RTL nodes "
                 "(out of context, xc7z020)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "estimator_audit_vivado.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs")
    p.add_argument("--recipe", nargs="*", default=[])
    p.add_argument("--points")
    p.add_argument("--acc")
    p.add_argument("--idle", nargs="*", default=[], help="outputs of energy_search.py idle")
    p.add_argument("--sweep", help="output of threshold_sweep.py")
    p.add_argument("--budget", help="output of budget_sweep.py")
    p.add_argument("--cands", help="folder with cand_<bits>.json QAT runs (needs --points)")
    p.add_argument("--latency", help="output of energy_search.py idle --latency")
    p.add_argument("--protocol", help="output of cand_table.py --json")
    p.add_argument("--protocol-model", help="which accuracy model label to plot (default: first)")
    p.add_argument("--audit", help="audit.json of audit_rtl_ops.py vivado (needs --infeasible)")
    p.add_argument("--calib", help="calib_v2.json with the leave-one-design-out results (calib_v2.py fit)")
    p.add_argument("--infeasible", help="infeasible.json of prove_infeasible.py vivado")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.audit:
        fig_audit(args.audit, args.infeasible, args.out)
    if args.calib:
        fig_calibration(args.calib, args.out)
    if args.protocol:
        fig_protocol(args.protocol, args.out, args.protocol_model)
        if args.points:
            fig_h3_pairs(np.load(args.points)["rows"], args.protocol, [
                ("margin 2, 1-rep model", "88242488", "44242888"),
                ("margin 3, 1-rep model", "84242444", "44242428"),
                ("margin 2, 2/3-rep model", "84242448", "44242448"),
                ("margin 3, 2-rep model", "84242248", "42442488")], args.out)
    if args.latency:
        fig_clock_timeline(args.out)
        sets = json.load(open(args.latency))
        for pname in [k for k in ("nominal", "z2_anchor", "z2_measured") if k in sets]:
            fig_latency(args.latency, args.out, pname)
    for f in args.idle:
        fig_idle(f, args.out)
    if args.sweep:
        fig_sweep(args.sweep, args.out)
    if args.budget:
        fig_budget(args.budget, args.out)
    if args.runs:
        fig_uniform(args.runs, args.out)
        fig_sensitivity(args.runs, args.out)
    if args.recipe:
        fig_recipe(args.recipe, args.out)
    if args.points and args.cands:
        rows = np.load(args.points)["rows"]
        fig_candidates(rows, args.cands, {
            "4,4,4,4,4,4,8,8": "A' (84.46%), also B2/B3/B4",
            "4,8,4,8,4,8,4,8": "B1 uniform W4A8",
            "8,4,8,4,8,4,8,4": "B1 uniform W8A4",
            "8,4,2,4,8,4,8,8": "A' fixed-rate (84.46%)",
            "8,8,2,4,2,4,8,8": "A' (83.46%)",
            "4,4,2,4,2,8,8,8": "BOPs proxy (83.46%)",
            "8,4,2,4,2,4,4,4": "A' (82.46%)",
            "4,4,2,4,2,4,2,8": "BOPs proxy (82.46%)"}, args.out)
    if args.points and not args.cands:
        rows = np.load(args.points)["rows"]
        fig_hw(rows, args.out)
        if args.acc:
            acc = json.load(open(args.acc))["acc"]
            for pname in POWER:
                fig_energy(rows, acc, args.out, pname)
    print("figures in", args.out, os.listdir(args.out))


if __name__ == "__main__":
    main()
