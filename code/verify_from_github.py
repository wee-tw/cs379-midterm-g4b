"""Recompute the headline numbers from the data on GitHub and compare them with the committed results.

  1. Main result tables (timing_safe.py table, 9 PYNQ clock classes, 5 seeds), without and with the
     +10 % routing margin -> results/search/main_table_tsr_v1218_m0{,.1}.json
  2. Plan D checks (plan_d/plan_d_checks.py) -> results/plan_d/plan_d_checks.txt, rq2_729_energies.csv

Needs Python 3 with numpy and scipy. Nothing has to be unpacked first. Run from anywhere:

  python code/verify_from_github.py
"""

import glob
import json
import math
import os
import subprocess
import sys
import tempfile

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
CLASSES = ("125", "111.11", "100", "90.91", "83.33", "76.92", "71.43", "66.67", "50")


def same(a, b, path=""):
    """First difference between two JSON values (floats to 1e-9 relative), or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return path + "/" + k + " only in one file"
            d = same(a[k], b[k], path + "/" + k)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path + " length %d vs %d" % (len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            return None
    elif a == b:
        return None
    return "%s: %r vs %r" % (path, a, b)


def run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    r = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode


def main():
    ok = True
    logs = sorted(glob.glob(os.path.join(ROOT, "results", "accuracy", "train_logs", "exp*")))
    with tempfile.TemporaryDirectory() as tmp:
        for tag, name in (("tsr_v1218", "main_table_tsr_v1218_m0.json"), ("tsr_v1218_m0.1", "main_table_tsr_v1218_m0.1.json")):
            out = os.path.join(tmp, name)
            pts = ["%s=%s" % (c, os.path.join(CODE, "cost", "points_%s_f%s.npz" % (tag, c))) for c in CLASSES]
            rc = run([sys.executable, "timing_safe.py", "table", "--acc", os.path.join(CODE, "cost", "acc_model_3rep.json"),
                      "--sweep", os.path.join(CODE, "cost", "threshold_sweep_%s_merged.json" % tag),
                      "--points"] + pts + ["--logs"] + logs + ["--out", out])
            ref = os.path.join(ROOT, "results", "search", name)
            diff = "run failed" if rc else same(json.load(open(ref, encoding="utf-8")), json.load(open(out, encoding="utf-8")))
            print("%-36s %s" % (name, "same as results/search/" if diff is None else "DIFFERENT: " + diff))
            ok &= diff is None

        rc = run([sys.executable, os.path.join("plan_d", "plan_d_checks.py"), "--points-dir", os.path.join(CODE, "cost"), "--out", tmp])
        for name in ("plan_d_checks.txt", "rq2_729_energies.csv"):
            ref = open(os.path.join(ROOT, "results", "plan_d", name), encoding="utf-8").read().replace("\r\n", "\n")
            new = "" if rc else open(os.path.join(tmp, name), encoding="utf-8").read().replace("\r\n", "\n")
            good = rc == 0 and ref == new
            print("%-36s %s" % (name, "same as results/plan_d/" if good else "DIFFERENT"))
            ok &= good
    print("ALL MATCH" if ok else "SOME RESULTS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
