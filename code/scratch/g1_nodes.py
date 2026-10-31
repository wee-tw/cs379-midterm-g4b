import json, sys
t = json.load(open(sys.argv[1]))
tab = t["table"] if "table" in t else t
keys = [k for k in tab if k.startswith("dsp|12500|0|")]
print("G1 dsp keys at T=12500:", keys[:12])
for k in keys:
    if k.split("|")[3] in ("4", "8") and k.split("|")[4] == "4":
        print("==", k)
        for n in tab[k]["nodes"]:
            v = n.get("variants") or {}
            best = min(v.values(), key=lambda r: r.get("LUT", 0)) if isinstance(v, dict) and v else n.get("auto", {})
            print(f"   {n['op'] if 'op' in n else n.get('op_type', '?'):34s} cyc {n['cycles']:7d}  auto LUT {n['auto'].get('LUT', 0):7.0f} DSP {n['auto'].get('DSP', 0):4.0f}  BRAM {n['auto'].get('BRAM_18K', 0):5.1f}")
