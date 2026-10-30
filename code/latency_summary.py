"""Summarise energy_search.py idle --latency output (model estimates).

Per power set and rate: the II-based gated optimum (the earlier H2' model) against the
latency-aware policies, where frames are handled one at a time:
  streaming  lowclk (clock running, lowest feasible clock) vs gated_lat (halt when the
             pipeline drains), B5 = fastest design with halts
  deadline   each frame must finish before the next one arrives: dl_running vs dl_gated,
             with B5 (fastest II) and B4L (lowest latency) halted between frames

  python latency_summary.py idle_lat.json
"""

import json
import sys


def e(x):
    return x["E_mJ"] if isinstance(x, dict) else x


def pct(a, b):
    return None if a is None or b is None else round(100 * (a / b - 1), 1)


def main():
    d = json.load(open(sys.argv[1]))
    for p, res in d.items():
        print(f"== {p}")
        for R, r in res.items():
            if R == "continuous":
                continue
            low, gl = e(r["lowclk"]["A_prime"]), e(r["gated_lat"]["A_prime"])
            jit, bound = e(r["jit"]["A_prime"]), e(r["gated"]["A_prime"])
            dlr, dlg = e(r["dl_running"]["A_prime"]), e(r["dl_gated"]["A_prime"])
            dlj = e(r["dl_jit"]["A_prime"])
            b5s = r["gated_lat"]["B4_latency_mJ"]
            b5d, b4l = r["dl_gated"]["B4_latency_mJ"], r["dl_gated"]["B4L_min_latency_mJ"]
            a, aj = r["dl_gated"]["A_prime"], r["jit"]["A_prime"]
            best = min(low, gl, jit)
            print(f"  R={R:>5}: streaming: jit {jit:.4f} ({aj['bits']} cyc {aj['cycles']} "
                  f"f {aj['f_MHz']:.3f}) = II-bound +{pct(jit, bound)}%; lowclk +{pct(low, best)}%, "
                  f"halt +{pct(gl, best)}%, B5 +{pct(b5s, best)}% | deadline: "
                  + ("infeasible" if dlg is None else
                     f"gated {dlg:.4f} {a['bits']} cyc {a['cycles']}, jit {pct(dlj, dlg):+}%, "
                     f"F_EXT running +{pct(dlr, dlg)}%, B5 +{pct(b5d, dlg)}%, B4L +{pct(b4l, dlg)}%"))


if __name__ == "__main__":
    main()
