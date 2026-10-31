"""Post-synthesis timing of every synthesized MVAU_hls (implementation_notes.md 12.35).

Reads the export_design -flow syn reports (Vitis HLS 2022.2, synth_design only, so the
periods are post-synthesis estimates) and the generated top_*.cpp of each case, and prints
the period by input folding (SF = MW / SIMD) and SIMD. timing_safe.py turns the result
into the MVAU_hls clock classes.

  python hls_timing_scan.py D:/vd D:/vd2 D:/vd3 D:/vd4 <S>/audit_hls <S>/audit_hls_finn <S>/audit_hls_trained
"""

import glob
import os
import re
import sys


def scan(roots):
    rows = []
    for root in roots:
        for rpt in glob.glob(os.path.join(root, "*", "export_syn.rpt")):
            d = os.path.dirname(rpt)
            cpp = glob.glob(os.path.join(d, "top_*.cpp"))
            if not cpp:
                continue
            src = open(cpp[0], encoding="utf-8", errors="ignore").read()
            if "Matrix_Vector_Activate" not in src:
                continue
            t = open(rpt, encoding="utf-8", errors="ignore").read()
            i = t.find("RTL Synthesis Timing Summary")
            tgt = re.search(r"Target\s*\|\s*([\d.]+)", t[i:i + 700])
            ach = re.search(r"Post-(?:Route|Synthesis)\s*\|\s*([\d.]+)", t[i:i + 700])
            if not (tgt and ach):
                continue
            g = lambda n: int(re.search(r"#define %s1 (\d+)" % n, src).group(1))
            inp = re.search(r"Slice<ap_(u?)int<(\d+)>>", src)
            res = re.search(r"ap_resource_(\w+)\(\)", src)
            rows.append(dict(root=os.path.basename(root), case=os.path.basename(d), MW=g("MW"),
                             MH=g("MH"), SIMD=g("SIMD"), PE=g("PE"), W=g("WP"),
                             A=int(inp.group(2)) if inp else -1, TMEM=g("TMEM"),
                             res=res.group(1) if res else "?", target=float(tgt.group(1)),
                             period=float(ach.group(1))))
    return rows


def main():
    rows = scan(sys.argv[1:])
    print(f"MVAU_hls synthesis runs with timing: {len(rows)}")
    print("| target | SF == 1 | SIMD | runs | over target | worst (ns) |")
    print("|---|---|---|---|---|---|")
    for tgt in sorted({r["target"] for r in rows}):
        for sf1 in (True, False):
            for lo, hi in [(1, 27), (28, 47), (48, 9999)]:
                s = [r for r in rows if r["target"] == tgt and (r["MW"] == r["SIMD"]) == sf1
                     and lo <= r["SIMD"] <= hi]
                if s:
                    print(f"| {tgt:g} ns | {sf1} | {lo}-{hi} | {len(s)} | "
                          f"{sum(r['period'] > tgt for r in s)} | {max(r['period'] for r in s):.2f} |")


if __name__ == "__main__":
    main()
