import json, sys
d = json.load(open(sys.argv[1]))
for p, r in d.items():
    print("==", p)
    for reg, res in [("cont", r["continuous"])] + [("R=" + k, v) for k, v in r["fixed_rate"].items()]:
        for k in ("A_prime", "B1_uniform", "B4_latency"):
            if k not in res: continue
            x = res[k]
            e = x.get("E_fixed_mJ", x["E_continuous_mJ"]) if reg != "cont" else x["E_continuous_mJ"]
            print(f"  {reg:7s} {k:11s} E={e:8.3f} f={x['f_MHz']:6.2f} cyc={x['cycles']:7d} U={x['U']:.3f} "
                  f"LUT={x['LUT']:6d} BRAM={x['BRAM18']:5.1f} DSP={x['DSP']:3d} mode={x['mode']:8s} bits={x['bits']}")
