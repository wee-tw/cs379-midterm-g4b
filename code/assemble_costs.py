"""Assemble whole-network FINN cost estimates for ResNet-8 from the per-group tables.

  python assemble_costs.py validate --tables tables.json --n 12

`validate` re-runs the FINN estimate flow directly on random configurations that are
not in the covering set, and checks that the assembled totals (FINN's "auto" choices)
and the bottleneck cycles match exactly.

The library part (used by energy_search.py):
  assemble(table, bits, mode, T)      -> list of node records (with group id)
  choose_variants(nodes, budget, w)   -> per-node implementation choice that minimises
                                         the weighted utilisation inside the budget
"""

import argparse
import json
import random

import numpy as np

from cost_tables import contexts, covering_configs

RES = ("LUT", "BRAM_18K", "DSP")
# Zynq-7020 (PYNQ-Z2) device totals
DEVICE = {"LUT": 53200, "BRAM_18K": 280, "DSP": 220}


def load_tables(path):
    d = json.load(open(path))
    return d["table"], d["T_GRID"]


def assemble(table, bits, mode, T):
    nodes = []
    for g, (w, a, p) in enumerate(contexts(bits)):
        rec = table.get(f"{mode}|{T}|{g}|{w}|{a}|{p}")
        if rec is None:
            return None
        nodes += rec["nodes"]
    return nodes


def options(node):
    """Implementation options of one node: FINN's variants if recorded, else auto."""
    if node.get("variants"):
        return node["variants"]
    return [dict(attrs={}, **node["auto"])]


def util(r, w):
    return sum(w[k] * r[k] / DEVICE[k] for k in RES)


def choose_variants(nodes, budget, w):
    """Minimise sum of weighted utilisation subject to the LUT/BRAM/DSP budget.

    Separable optimum first (each node takes its own cheapest option); only if that
    breaks the budget, solve the multiple-choice problem as a small MILP.
    Returns (choice index per node, totals) or (None, None) if infeasible."""
    opts = [options(n) for n in nodes]
    pick = [int(np.argmin([util(o, w) for o in os_])) for os_ in opts]
    tot = {k: sum(opts[i][pick[i]][k] for i in range(len(nodes))) for k in RES}
    if all(tot[k] <= budget[k] for k in RES):
        return pick, tot
    from scipy.optimize import Bounds, LinearConstraint, milp
    idx = [(i, j) for i, os_ in enumerate(opts) for j in range(len(os_))]
    c = np.array([util(opts[i][j], w) for i, j in idx])
    A_one = np.zeros((len(nodes), len(idx)))
    for col, (i, j) in enumerate(idx):
        A_one[i, col] = 1
    A_res = np.array([[opts[i][j][k] for (i, j) in idx] for k in RES])
    cons = [LinearConstraint(A_one, 1, 1),
            LinearConstraint(A_res, -np.inf, [budget[k] for k in RES])]
    # presolve off: scipy 1.10.1's HiGHS presolve crashes (access violation) on some of these
    # instances, e.g. bits 8,8,4,2,2,4,2,2 / lut / T=25000 at LUT 0.7, BRAM 0.6 derating
    res = milp(c, constraints=cons, integrality=np.ones(len(idx)), bounds=Bounds(0, 1),
               options={"presolve": False})
    if not res.success:
        return None, None
    x = np.round(res.x).astype(int)
    pick = [0] * len(nodes)
    for col, (i, j) in enumerate(idx):
        if x[col]:
            pick[i] = j
    tot = {k: sum(opts[i][pick[i]][k] for i in range(len(nodes))) for k in RES}
    return pick, tot


def auto_totals(nodes):
    return {k: sum(n["auto"][k] for n in nodes) for k in RES}


def validate(args):
    import tempfile
    import cost_tables as ct
    table, grid = load_tables(args.tables)
    cover = {tuple(c) for c in covering_configs()}
    rng = random.Random(1)
    import itertools
    allcfg = [list(c) for c in itertools.product((2, 4, 8), repeat=8)]  # Python ints for Brevitas
    cands = [c for c in allcfg if tuple(c) not in cover]
    picks = rng.sample(cands, args.n)
    bad = 0
    for bits in picks:
        for mode, (standalone, wwidth) in ct.MODES.items():
            m = ct.hw_model(bits, standalone, tempfile.mkdtemp())
            for T in rng.sample(grid, 3):
                direct = ct.fold_and_estimate(m, T, wwidth)
                nodes = assemble(table, bits, mode, T)
                d_tot = {k: sum(r["auto"][k] for r in direct) for k in RES}
                a_tot = auto_totals(nodes)
                d_cyc = max(r["cycles"] for r in direct)
                a_cyc = max(r["cycles"] for r in nodes)
                ok = d_tot == a_tot and d_cyc == a_cyc
                bad += not ok
                print(bits, mode, T, "direct", d_tot, d_cyc, "assembled", a_tot, a_cyc,
                      "OK" if ok else "MISMATCH", flush=True)
    print("mismatches:", bad)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["validate"])
    p.add_argument("--tables", required=True)
    p.add_argument("--n", type=int, default=12)
    validate(p.parse_args())
