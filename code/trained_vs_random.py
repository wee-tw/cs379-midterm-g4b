"""Does it matter that the cost tables are built from a randomly initialised ResNet-8?

FINN's folding, weight memories and DSP counts depend only on the shapes and bit widths, but
MinimizeAccumulatorWidth sizes the accumulators (and so the threshold widths) from the actual
weights, and the thresholds come from the trained BatchNorm. This rebuilds a few configurations
with their trained checkpoints and compares FINN's per-node estimates with the random-init model
used by cost_tables.py, at the same folding.

  python trained_vs_random.py --ckpt 44444488=exp8/ft30_cand_44444488.pt ...
"""

import argparse
import tempfile

import cost_tables as ct

RES = ("LUT", "BRAM_18K", "DSP")


def estimates(bits, standalone, T, wwidth, ckpt):
    from qonnx.custom_op.registry import getCustomOp
    m = ct.hw_model(bits, standalone, tempfile.mkdtemp(), ckpt=ckpt)
    rows = ct.fold_and_estimate(m, T, wwidth)
    out = {}
    for r in rows:
        out[r["name"]] = r
    accs = {}
    for n in m.graph.node:
        if n.op_type.startswith("Thresholding") or n.op_type.startswith("MVAU"):
            inst = getCustomOp(n)
            if "inputDataType" in inst.get_nodeattr_types():
                accs[n.name] = inst.get_nodeattr("inputDataType")
    return out, accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="bits=checkpoint.pt")
    p.add_argument("--T", type=int, nargs="+", default=[100000, 25000])
    args = p.parse_args()
    for spec in args.ckpt:
        bits_s, path = spec.split("=", 1)
        bits = list(map(int, bits_s))
        for mode, (standalone, wwidth) in (("dsp", ct.MODES["dsp"]), ("lut_finn", ct.MODES["lut_finn"])):
            for T in args.T:
                r_rand, _ = estimates(bits, standalone, T, wwidth, None)
                r_tr, _ = estimates(bits, standalone, T, wwidth, path)
                tot = {k: (sum(r["auto"][k] for r in r_rand.values()),
                           sum(r["auto"][k] for r in r_tr.values())) for k in RES}
                cyc = (max(r["cycles"] for r in r_rand.values()), max(r["cycles"] for r in r_tr.values()))
                diffs = [(n, r_rand[n]["auto"]["LUT"], r_tr[n]["auto"]["LUT"]) for n in r_rand
                         if n in r_tr and abs(r_rand[n]["auto"]["LUT"] - r_tr[n]["auto"]["LUT"]) > 0.5]
                print(f"{bits_s} {mode:8s} T {T:6d} | cycles {cyc[0]} / {cyc[1]} | " + " | ".join(
                    f"{k} random {a:.0f} trained {b:.0f} ({100 * (b / a - 1) if a else 0:+.1f}%)"
                    for k, (a, b) in tot.items()) + f" | nodes with LUT changes: {len(diffs)}", flush=True)
                for n, a, b in diffs[:4]:
                    print(f"      {n}: LUT {a:.0f} -> {b:.0f}")


if __name__ == "__main__":
    main()
