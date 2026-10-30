"""Markdown summary of a run directory: every job's final accuracy, plus the per-block
sensitivity matrix if present.

  python summarize_runs.py exp3 [exp2 ...]
"""

import glob
import itertools
import json
import os
import sys

PAIRS = list(itertools.product((2, 4, 8), repeat=2))


def final(path):
    r = json.load(open(path))
    return r["history"][-1]["test_acc"] if "history" in r else None


for d in sys.argv[1:]:
    print(f"\n### {os.path.basename(d.rstrip('/'))}\n")
    print("| run | final top-1 (%) | epochs | wall (min) |")
    print("|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) == "jobs.json":
            continue
        r = json.load(open(f))
        if "history" not in r:
            continue
        print(f"| {os.path.basename(f)[:-5]} | {r['history'][-1]['test_acc']:.2f} | "
              f"{len(r['history'])} | {r['wall_sec'] / 60:.1f} |")
    base_f = os.path.join(d, "sens_base.json")
    if os.path.exists(base_f):
        base = final(base_f)
        print(f"\nSensitivity (Δ vs control {base:.2f}%):\n")
        print("| group | " + " | ".join(f"W{w}A{a}" for w, a in PAIRS) + " |")
        print("|---|" + "---|" * len(PAIRS))
        for g in range(4):
            cells = []
            for w, a in PAIRS:
                if (w, a) == (4, 4):
                    cells.append("0 (base)")
                    continue
                f = os.path.join(d, f"sens_g{g + 1}_w{w}a{a}.json")
                cells.append(f"{final(f) - base:+.2f}" if os.path.exists(f) else "–")
            print(f"| G{g + 1} | " + " | ".join(cells) + " |")
