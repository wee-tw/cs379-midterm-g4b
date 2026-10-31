"""Totals of the synthesized designs under the v1 and v2 tables, same variant choices (v1's)."""
import json, os, sys
from assemble_costs import assemble, choose_variants, load_tables, options
from energy_search import DERATE, DEVICE, RES, UTIL_W
S = sys.argv[1]
# --v2-choices: the designs were generated with the v2 tables' variant choices (D:/vd3)
v2c = "--v2-choices" in sys.argv
dirs = [a for a in sys.argv[2:] if a != "--v2-choices"]
t1, _ = load_tables(f"{S}/cost/tables_calov.json"); t2, _ = load_tables(f"{S}/cost/tables_calv2.json")
budget = {k: DEVICE[k] * DERATE[k] for k in RES}
print("budget", budget)
for dd in dirs:
    cases = json.load(open(f"{dd}/cases.json")); res = json.load(open(f"{dd}/results.json"))
    for tag in sorted({c["design"] for c in cases}):
        bits, mode, T = tag.split("_")[0], "_".join(tag.split("_")[1:-1]), int(tag.rsplit("_", 1)[1])
        n1 = assemble(t1, list(map(int, bits)), mode, T); n2 = assemble(t2, list(map(int, bits)), mode, T)
        if mode == "lut_finn":
            o1 = [n["auto"] for n in n1]; o2 = [n["auto"] for n in n2]
        else:
            pick, _ = choose_variants(n2 if v2c else n1, budget, UTIL_W)
            o1 = [options(n)[p] for n, p in zip(n1, pick)]; o2 = [options(n)[p] for n, p in zip(n2, pick)]
        cs = [c for c in cases if c["design"] == tag]
        ok = [res.get(c["case"], {}) for c in cs]
        missing = sum(1 for v in ok if v.get("returncode") != 0)
        syn = {k: sum((v.get(k) or 0) for v in ok) for k in ("LUT", "BRAM18", "DSP")}
        p1 = {k: sum(o[k] for o in o1) for k in ("LUT", "BRAM_18K", "DSP")}
        p2 = {k: sum(o[k] for o in o2) for k in ("LUT", "BRAM_18K", "DSP")}
        print(f"{os.path.basename(dd)}:{tag:26s} synth LUT {syn['LUT']:6.0f} BRAM18 {syn['BRAM18']:4.0f} DSP {syn['DSP']:4.0f}"
              f" | v1 LUT {p1['LUT']:6.0f} ({100*(p1['LUT']/syn['LUT']-1):+.1f}%) | v2 LUT {p2['LUT']:6.0f} ({100*(p2['LUT']/syn['LUT']-1):+.1f}%)"
              f" BRAM {p2['BRAM_18K']:4.0f} DSP {p2['DSP']:4.0f}" + (f" [{missing} cases missing]" if missing else ""))
