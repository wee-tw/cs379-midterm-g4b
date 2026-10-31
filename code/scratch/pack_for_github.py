"""Copy 1133311's local experiment data into the repo so the team can get it from GitHub.

  * code/cost/: every cost table and design-point file from the working folder (except test and
    smoke runs and five byte-identical reproduction copies). Point files (.npz) are re-saved with
    np.savez_compressed (same names, np.load reads them unchanged, checked array by array); the
    JSON tables are stored as .json.gz (checked byte for byte; code/cost/unpack.py restores them).
  * results/checkpoints/: the exp13-exp18 training runs (and their 1-epoch smoke runs).
  * results/raw/*.tar.gz: synthesis reports, audit folders and training logs, text files only
    (no Vivado checkpoints or project databases); results/raw/MANIFEST.csv lists every file.

  python code/scratch/pack_for_github.py --scratch <working folder> --redact <host name> [--only cost|ckpt|raw]
The host name in tool report headers is replaced with <host> (the repo is public); nothing else changes.
"""

import argparse
import csv
import gzip
import hashlib
import io
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
IDENTICAL_REPRO = {"tables_calov_repro.json", "tables_corr_hi_repro.json", "tables_thr_hi_repro.json",
                   "tables_viv_hlsov_repro.json", "tables_viv_repro.json"}
TEXT = {".rpt", ".log", ".json", ".jsonl", ".tcl", ".csv", ".txt", ".xdc", ".sv", ".v", ".vh", ".vhd",
        ".vhdl", ".dat", ".npy", ".h", ".hpp", ".cpp", ".py", ".sh", ".ys", ".out", ".md"}
REPORTS = {".rpt", ".log", ".json", ".tcl", ".csv", ".txt", ".xdc"}
MAX_FILE = 20 * 2**20


def sha(path, opener=open):
    h = hashlib.sha256()
    with opener(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skip_cost(name):
    return "_test" in name or "smoke" in name or name.startswith("fake_") or name in IDENTICAL_REPRO


def pack_npz(src, dst):
    try:
        with np.load(src, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
    except ValueError:
        with np.load(src, allow_pickle=True) as z:
            arrs = {k: z[k] for k in z.files}
    tmp = dst[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    back = np.load(tmp, allow_pickle=True)
    assert sorted(back.files) == sorted(arrs), src
    for k, a in arrs.items():
        b = back[k]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (src, k)
    back.close()
    os.replace(tmp, dst)
    return dst


def pack_json(src, dst):
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    assert sha(src) == sha(tmp, gzip.open), src
    os.replace(tmp, dst)
    return dst


def cost_jobs(src_dir, dst_dir):
    jobs = []
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src) or skip_cost(name):
            continue
        if name.endswith(".npz"):
            jobs.append(("npz", src, os.path.join(dst_dir, name)))
        elif name.endswith(".json") and (name.startswith("tables") or os.path.getsize(src) > 2**20):
            jobs.append(("json", src, os.path.join(dst_dir, name + ".gz")))
        else:
            jobs.append(("copy", src, os.path.join(dst_dir, name)))
    return jobs


def run_job(job):
    kind, src, dst = job
    size = os.path.getsize(src)
    if kind == "npz":
        pack_npz(src, dst)
    elif kind == "json":
        pack_json(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst, size, os.path.getsize(dst), kind


def rule_text(rel, size):
    return os.path.splitext(rel)[1].lower() in TEXT and size <= MAX_FILE and ".Xil" not in rel


def rule_case_reports(rel, size):
    """Vivado working folders: only the reports, scripts and logs at the top of each case."""
    return rel.count("/") <= 1 and os.path.splitext(rel)[1].lower() in REPORTS and size <= MAX_FILE


def rule_small(rel, size):
    return rule_text(rel, size) and size <= 2**20


def rule_training(rel, size):
    """Training logs also keep their small checkpoints (.pt), arrays (.npz) and ONNX models."""
    return (rule_text(rel, size) or os.path.splitext(rel)[1].lower() in (".pt", ".npz", ".onnx")) and size <= MAX_FILE


def rule_builds(rel, size):
    """FINN builds: reports and logs, not the intermediate ONNX models."""
    return rule_text(rel, size) and "intermediate_models/" not in rel and "_tmp/" not in rel


LEFT_OUT_TOP = {"oa.json", "osscad.json", "numpy1133.json", "tiny07_tree.json", "tiny10_tree.json",
                "tiny11_tree.json", "install_torch.log", "b3test.json"}


def rule_toplevel_outputs(rel, size):
    """Logs and results written at the top of the working folder (not literature or tool lookups)."""
    return ("/" not in rel and os.path.splitext(rel)[1].lower() in (".log", ".json", ".jsonl", ".csv", ".out")
            and rel not in LEFT_OUT_TOP and size <= MAX_FILE)


RULES = {f.__name__: f for f in (rule_text, rule_case_reports, rule_small, rule_training, rule_builds,
                                 rule_toplevel_outputs)}
BINARY = {".npy", ".pt", ".npz", ".onnx", ".dat"}


def raw_groups(scratch):
    d = lambda *names: [("D:/" + n, n) for n in names]  # noqa: E731
    s = lambda *names: [(os.path.join(scratch, n), n) for n in names]  # noqa: E731
    return [
        ("synth_nodes_vd", rule_text, d("vd", "vd2", "vd3", "vd4", "vd5", "vd6", "vd7", "vd8", "vdev", "vt_test")),
        ("synth_dwc_vdw", rule_text, d("vdw", "vdw_narrow", "vdw_narrow_m5", "vdw_smoke", "vdw_smoke2")),
        ("synth_rtl_checks", rule_text, d("va", "vi", "vq", "vr", "vr3", "vr6", "vr8a", "vr8b", "vr8c", "vr8d",
                                           "vrc", "vrdev", "vrs", "vs", "vt", "vt2", "vwt")),
        ("vivado_nodes_vw", rule_case_reports, d("vw", "vw2", "vw3", "vw4", "vw5", "vw6", "vw7", "vw8")),
        ("vivado_dwc_vww", rule_case_reports, d("vww", "vww_narrow", "vww_narrow_m5", "vww_smoke")),
        ("audit_rtl", rule_text, s("audit", "audit_thr_trig", "infeasible", "route", "rtl_timing", "synth",
                                    "synth_one", "synth_prims")),
        ("audit_hls", rule_text, s("audit_hls", "audit_hls_finn", "audit_hls_trained")),
        ("training_logs", rule_training,
         s("bn", "eval", "robust_logs", "runs", "digest_f143", "digest_lut07", "digest_lut09", "smoke_h1", "rn8")),
        ("finn_builds_calibration", rule_builds,
         s("cnv_calib", "fe_util")),
        ("clock_class_checks", rule_small, s("ts_test")),
        ("working_folder_outputs", rule_toplevel_outputs, [(scratch, "work")]),
        ("board_mock_sessions", rule_text, s("mock_sessions", "sessions_test", "sessions_now", "sessions_now2",
                                              "sessions_py36", "sessions_py36b")),
    ]


def pack_group(args):
    name, rule_name, sources, out_dir, redact = args
    rule = RULES[rule_name]
    out = os.path.join(out_dir, name + ".tar.gz")
    rows, skipped = [], 0
    with tarfile.open(out + ".tmp", "w:gz", compresslevel=9) as tar:
        for src, label in sources:
            if not os.path.isdir(src):
                continue
            for root, dirs, files in os.walk(src):
                dirs[:] = [] if rule_name == "rule_toplevel_outputs" else sorted(x for x in dirs if x != ".Xil")
                for f in sorted(files):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, src).replace(os.sep, "/")
                    size = os.path.getsize(p)
                    if rule(rel, size):
                        data = open(p, "rb").read()
                        if redact and os.path.splitext(rel)[1].lower() not in BINARY:
                            data = data.replace(redact.encode(), b"<host>")
                        info = tarfile.TarInfo(label + "/" + rel)
                        info.size, info.mtime = len(data), int(os.path.getmtime(p))
                        tar.addfile(info, io.BytesIO(data))
                        rows.append((name + ".tar.gz", label + "/" + rel, size))
                    else:
                        skipped += 1
    os.replace(out + ".tmp", out)
    return name, rows, skipped, os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--only", choices=("cost", "ckpt", "raw"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--redact", default="", help="host name to replace with <host> in the text files")
    args = ap.parse_args()

    if args.only in (None, "cost"):
        dst = os.path.join(ROOT, "code", "cost")
        jobs = cost_jobs(os.path.join(args.scratch, "cost"), dst)
        v0 = os.path.join(dst, "v0")
        for name in ("tables.json",):
            jobs.append(("json", os.path.join(v0, name), os.path.join(v0, name + ".gz")))
        for name in ("points.npz", "latency.npz"):
            src = os.path.join(v0, name)
            if os.path.exists(src):
                jobs.append(("npz", src, src))
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(run_job, jobs))
        a, b = sum(r[1] for r in res), sum(r[2] for r in res)
        print("cost: %d files, %.0f MB -> %.0f MB" % (len(res), a / 2**20, b / 2**20))

    if args.only in (None, "ckpt"):
        dst = os.path.join(ROOT, "results", "checkpoints")
        for n in ("exp13", "exp14", "exp15", "exp16", "exp17", "exp18", "exp13_smoke", "exp14_smoke",
                  "exp15_smoke", "exp16_smoke", "exp17_smoke", "exp18_smoke"):
            shutil.copytree(os.path.join(args.scratch, n), os.path.join(dst, n), dirs_exist_ok=True)
        print("checkpoints copied")

    if args.only in (None, "raw"):
        out_dir = os.path.join(ROOT, "results", "raw")
        os.makedirs(out_dir, exist_ok=True)
        groups = [(n, r.__name__, s, out_dir, args.redact) for n, r, s in raw_groups(args.scratch)]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(pack_group, groups))
        with open(os.path.join(out_dir, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["archive", "path", "bytes"])
            for name, rows, skipped, size in res:
                wr.writerows(rows)
        for name, rows, skipped, size in res:
            print("%-26s %6d files %8.1f MB -> %6.1f MB (left out %d)"
                  % (name, len(rows), sum(r[2] for r in rows) / 2**20, size / 2**20, skipped))


if __name__ == "__main__":
    main()
