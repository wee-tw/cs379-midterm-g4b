"""H3, constructive part: which hardware-blind proxy would pick the energy-optimal bits?

BOPs count multiply bit-operations only. FINN also turns every quantised activation into a
thresholding step: each A-bit output element is compared against 2^A - 1 thresholds of
accumulator width, and each channel stores those thresholds. This script adds those terms
to BOPs and to weight memory, using only the layer shapes and the bit widths (no FINN run),
and checks every proxy against the energy model (estimates):
  * Kendall's tau with the best continuous energy of each bit configuration
  * the energy regret of the proxy's pick at each accuracy margin (min proxy among the
    configurations the accuracy model passes; best folding/clock for those bits)

  python proxy_finn.py --points cost/points.npz --acc cost/acc_model_3rep.json --out proxy_finn.json
"""

import argparse
import json
import math

import numpy as np
from scipy.stats import kendalltau

from energy_search import ALL_CONFIGS, F_SET, POWER, bops, energies, wmem_bits

# thresholding sites of the streamlined ResNet-8 (one per quantised activation):
# (name, output elements per frame, channels, index of A1..A4, weight group, input act
#  index or None for the 8-bit image, fan-in, residual add before the thresholds)
SITES = [("stem", 16384, 16, 0, 0, None, 27, False),
         ("s1c1", 16384, 16, 0, 0, 0, 144, False),
         ("add1", 16384, 16, 0, 0, 0, 144, True),
         ("s2c1", 8192, 32, 1, 1, 0, 144, False),
         ("add2", 8192, 32, 1, 1, 1, 288, True),
         ("s3c1", 4096, 64, 2, 2, 1, 288, False),
         ("add3", 4096, 64, 3, 2, 2, 576, True)]


def acc_width(b, w_g, a_in, fan_in, add):
    w, a = b[0::2], b[1::2]
    a_bits = 8 if a_in is None else a[a_in]
    return w[w_g] + a_bits + math.ceil(math.log2(fan_in)) + (1 if add else 0)


def threshold_terms(b):
    a = b[1::2]
    ops = bitops = mem = 0
    for _, elems, ch, a_out, w_g, a_in, fan_in, add in SITES:
        n_thr = 2 ** a[a_out] - 1
        width = acc_width(b, w_g, a_in, fan_in, add)
        ops += elems * n_thr
        bitops += elems * n_thr * width
        mem += ch * n_thr * width
    return ops, bitops, mem


def proxies(b):
    ops, bitops, mem = threshold_terms(b)
    return {"BOPs": bops(b), "weight bits": wmem_bits(b), "threshold compares": ops,
            "BOPs + threshold bit-ops": bops(b) + bitops,
            "weight + threshold bits": wmem_bits(b) + mem}


def best_energy(rows, pname):
    cfg = rows[:, 0].astype(int)
    _, e_c, _, _ = energies(rows, POWER[pname], max(F_SET))
    best = np.full(len(ALL_CONFIGS), np.inf)
    np.minimum.at(best, cfg, e_c)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--acc", required=True)
    p.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = np.load(args.points)["rows"]
    model = json.load(open(args.acc))
    acc = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    ref = model["reference_acc"]
    table = [proxies(b) for b in ALL_CONFIGS]
    names = list(table[0])
    out = {"acc_model": args.acc, "reference_acc": ref, "power": {}}
    for pname in ("nominal", "z2_anchor"):
        e = best_energy(rows, pname)
        ok = np.isfinite(e)
        res = {"kendall_tau": {}, "margins": {}}
        print(f"== {pname}: {ok.sum()} configurations with a feasible design")
        for n in names:
            v = np.array([t[n] for t in table], dtype=float)
            tau = kendalltau(v[ok], e[ok])[0]
            res["kendall_tau"][n] = round(float(tau), 3)
            print(f"  tau({n:26s}, energy) = {tau:+.3f}")
        for m in args.margins:
            thr = ref - m
            cand = np.where(ok & (acc >= thr))[0]
            a_best = cand[np.argmin(e[cand])]
            row = {"threshold": round(thr, 2), "n_pass": int(len(cand)),
                   "A_prime": "".join(map(str, ALL_CONFIGS[a_best])),
                   "E_A_prime_mJ": round(float(e[a_best]) * 1e3, 4), "proxies": {}}
            line = []
            for n in names:
                v = np.array([table[c][n] for c in cand], dtype=float)
                # ties: the more accurate configuration, as the BOPs baseline does
                pick = cand[np.lexsort((-acc[cand], v))[0]]
                regret = 100 * (e[pick] / e[a_best] - 1)
                row["proxies"][n] = {"bits": "".join(map(str, ALL_CONFIGS[pick])),
                                     "regret_pct": round(float(regret), 1)}
                line.append(f"{n.split(' ')[0] if n == 'BOPs' else n}: "
                            f"{''.join(map(str, ALL_CONFIGS[pick]))} {regret:+.1f}%")
            res["margins"][str(m)] = row
            print(f"  margin {m} (thr {thr:.2f}, {len(cand)} pass) A' {row['A_prime']} | "
                  + " | ".join(line))
        out["power"][pname] = res
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
