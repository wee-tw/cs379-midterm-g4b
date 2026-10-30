"""Joint vs sequential vs uniform as the accuracy threshold loosens (model estimates).

For each margin below the best predicted uniform accuracy, runs energy_search's
selections (A', B1-B4) and the idle-policy analysis (gated = race-to-halt) and prints
the energy of each baseline relative to A' in the same regime.

  python threshold_sweep.py --points points.npz --acc acc_model.json --out sweep.json
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import types

import numpy as np

import energy_search as es


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--acc", required=True)
    p.add_argument("--margins", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 8.0])
    p.add_argument("--powers", nargs="+", default=["nominal", "z2_anchor"])
    p.add_argument("--rates", nargs="+", default=["300", "3000"])
    p.add_argument("--latency", help="latency.npz: also report the latency-aware policies")
    p.add_argument("--lat-scale", type=float, default=0.6024)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = np.load(args.points)["rows"]
    a = json.load(open(args.acc))
    acc = [a["acc"][",".join(map(str, b))] for b in es.ALL_CONFIGS]
    out = {}
    tmp = tempfile.mkdtemp()
    for m in args.margins:
        thr = a["reference_acc"] - m
        acc_file = os.path.join(tmp, f"acc_{m}.json")
        json.dump({**a, "threshold": thr}, open(acc_file, "w"))
        idle_file = os.path.join(tmp, f"idle_{m}.json")
        with contextlib.redirect_stdout(io.StringIO()):
            es.cmd_idle(types.SimpleNamespace(points=args.points, acc=acc_file, out=idle_file,
                                              latency=args.latency, lat_scale=args.lat_scale))
        idle = json.load(open(idle_file))
        res = {"threshold": round(thr, 2), "n_pass": int(sum(x >= thr for x in acc))}
        for pname in args.powers:
            sel = es.search(rows, acc, thr, es.POWER[pname])
            c = sel["continuous"]
            ea = c["A_prime"]["E_continuous_mJ"]
            r = {"cont": {k: round(d["E_continuous_mJ"] / ea - 1, 4) for k, d in c.items()},
                 "cont_bits": {k: "".join(map(str, d["bits"])) for k, d in c.items()}}
            for R in args.rates:
                run = sel["fixed_rate"][R]
                er = run["A_prime"]["E_fixed_mJ"]
                r["run_" + R] = {k: round(d["E_fixed_mJ"] / er - 1, 4) for k, d in run.items()
                                 if d.get("E_fixed_mJ")}
                g = idle[pname][R]["gated"]
                eg = g["A_prime"]["E_mJ"]
                r["gated_" + R] = {"B1_uniform": round(g["B1_uniform"]["E_mJ"] / eg - 1, 4),
                                   **{k[:-3]: round(v / eg - 1, 4) for k, v in g.items()
                                      if k.endswith("_mJ") and v is not None}}
                r["gated_" + R + "_vs_run_A"] = round(eg / er - 1, 4)
                if args.latency:   # just-in-time clock, and one-frame-at-a-time halts (B5)
                    j, gl = idle[pname][R]["jit"], idle[pname][R]["gated_lat"]
                    ej = j["A_prime"]["E_mJ"]
                    r["jit_" + R] = {"A_prime_bits": "".join(map(str, j["A_prime"]["bits"])),
                                     "B1_uniform": round(j["B1_uniform"]["E_mJ"] / ej - 1, 4),
                                     **{k[:-3]: round(v / ej - 1, 4) for k, v in j.items()
                                        if k.endswith("_mJ") and v is not None},
                                     "B5_one_at_a_time": (round(gl["B4_latency_mJ"] / ej - 1, 4)
                                                          if gl["B4_latency_mJ"] else None)}
            res[pname] = r
        out[str(m)] = res
        print(f"== margin {m} (threshold {thr:.2f}, {res['n_pass']} configs pass)")
        for pname in args.powers:
            r = res[pname]
            print(f"  {pname:9s} cont vs A': " + ", ".join(f"{k.split('_')[0]} {100*v:+.1f}%"
                                                           for k, v in r["cont"].items())
                  + f"   bits A'={r['cont_bits']['A_prime']} B2={r['cont_bits'].get('B2_proxy')}")
            for R in args.rates:
                print(f"     R={R:>5s} running vs A': " + ", ".join(
                    f"{k.split('_')[0]} {100*v:+.1f}%" for k, v in r["run_" + R].items())
                    + f" | gated vs A'(gated): " + ", ".join(
                    f"{k.split('_')[0]} {100*v:+.1f}%" for k, v in r["gated_" + R].items())
                    + f" | A'(gated) vs A'(running) {100*r['gated_' + R + '_vs_run_A']:+.1f}%")
                if "jit_" + R in r:
                    jr = r["jit_" + R]
                    print(f"     R={R:>5s} jit: A'={jr['A_prime_bits']}; " + ", ".join(
                        f"{k.split('_')[0]} {100*v:+.1f}%" for k, v in jr.items()
                        if k != "A_prime_bits" and v is not None))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
