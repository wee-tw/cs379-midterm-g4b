"""Compare two timing_safe.py main tables margin by margin (A' pick, energy, baselines and gaps).
  python compare_tables.py <old table json> <new table json> [power]"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
old, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
p = sys.argv[3] if len(sys.argv) > 3 else "nominal"


def row(t, m):
    r = t[m][p]
    a = r["A_verified"]
    cells = [f"A' {a['bits']} {a['E_mJ']:.4f} mJ [{a.get('measured', '')}]"]
    for b in ("B1_measured", "B2_measured", "B3_measured"):
        d = r.get(b) or {}
        rel = d.get("rel_to_A")
        cells.append(f"{b[:2]} {d.get('bits')} " + ("-" if rel is None else f"{100 * rel:+.1f}%")
                     + (f" [{d.get('measured')}]" if d.get("measured") else ""))
    return cells


for m in sorted(new, key=float):
    o, n = row(old, m), row(new, m)
    flag = "" if o == n else "   <-- changed"
    print(f"margin {float(m):g}{flag}")
    for a, b in zip(o, n):
        print(f"   {a:62s} -> {b}" if a != b else f"   {a}")
