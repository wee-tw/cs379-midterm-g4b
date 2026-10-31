"""Energy of the final A' designs with the MW 576 RTL MVU narrowed to PE 64, SIMD 3 (notes 12.44).

The routed LUT of the layer goes from 2,492 (PE 16, SIMD 12; D:/vr6/d0_39_MVAU_rtl_5/route_util.rpt)
to 2,928 (PE 64, SIMD 3; D:/vr3/r576x64_pe64_s3_4x4/route_util.rpt); DSPs stay at 48 and the
cycle count of the design does not change. The 2 BRAM18 of the replay buffer that PE 64 (NF = 1)
drops are kept, to be conservative.

  python narrow_patch_energy.py --points $S/cost/points_tsr_v1218_f125.npz \
      --table ../../results/search/main_table_tsr_v1218_m0.json --lut-delta 436
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import energy_search as es  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="points npz of the 125 MHz class")
    ap.add_argument("--table", required=True, help="timing_safe.py table json")
    ap.add_argument("--lut-delta", type=int, default=436)
    ap.add_argument("--bram-delta", type=float, default=0.0)
    args = ap.parse_args()
    rows = np.load(args.points)["rows"]
    table = json.load(open(args.table))
    idx = {"".join(map(str, b)): i for i, b in enumerate(es.ALL_CONFIGS)}
    patched = {}
    for m in sorted(table, key=float):
        for p in ("nominal", "z2_anchor"):
            a = table[m][p]["A_verified"]
            key = (a["bits"], p)
            if key not in patched:
                mask = ((rows[:, 0].astype(int) == idx[a["bits"]]) & (rows[:, 1] == es.MODES.index(a["mode"]))
                        & (rows[:, 2] == a["T_target"]))
                row = rows[mask][:1].copy()
                e0 = es.energies(row, es.POWER[p], a["f_MHz"])[1][0] * 1e3
                row[0, 4] += args.lut_delta
                row[0, 5] += args.bram_delta
                e1 = es.energies(row, es.POWER[p], a["f_MHz"])[1][0] * 1e3
                patched[key] = (e0, e1)
                print(f"{a['bits']} {a['mode']} T{a['T_target']} {p:9s}: cycles {int(row[0, 3])}, "
                      f"E {e0:.4f} -> {e1:.4f} mJ ({100 * (e1 / e0 - 1):+.2f}%)")
    print(f"\nbaselines above A' (LUT {args.lut_delta:+d}, BRAM18 {args.bram_delta:+g}): before -> after")
    for m in sorted(table, key=float):
        out = []
        for p in ("nominal", "z2_anchor"):
            r = table[m][p]
            e0, e1 = patched[(r["A_verified"]["bits"], p)]
            out.append(f"{p}: " + "  ".join(
                f"{b[:2]} {100 * r[b]['rel_to_A']:+.1f} -> {100 * ((1 + r[b]['rel_to_A']) * e0 / e1 - 1):+.1f}%"
                for b in ("B1_measured", "B2_measured", "B3_measured") if r.get(b, {}).get("rel_to_A") is not None))
        print(f"margin {float(m):g} | " + " | ".join(out))


if __name__ == "__main__":
    main()
