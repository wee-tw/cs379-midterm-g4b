"""Do heuristic optimizers escape the estimator's harm? (implementation_notes.md 12.38; estimates)

dse_harm.py takes the biased model's exact optimum (exhaustive search). Design-space explorers
usually search heuristically, so here simulated annealing and a genetic algorithm search the
same space with the biased model as the objective (continuous energy at the best clock in F_SET;
points the biased model calls infeasible, or below the accuracy threshold, are rejected), within
a budget of objective evaluations. The optimizer's pick is then built: when the reference model
says it does not fit, the designer falls back through the designs the optimizer evaluated, best
(biased) first, one failed build each, as in dse_harm.py's build loop.

Space: 8 genes (W, A of the 4 groups, each 2/4/8), the mode and T; a neighbour changes one of
them by one step. Reported per margin over seeds: runs whose pick does not fit, failed builds,
regret of the first design that fits against the reference optimum, and whether the pick is the
biased model's exact optimum.

  python dse_heuristic.py --truth cost/points_reference_calv2.npz --biased cost/points_finn.npz \
      --acc cost/acc_model_3rep.json --margins 2 3 5 --seeds 20 --budget 2000
"""

import argparse
import json
import math
import random

import numpy as np

from energy_search import ALL_CONFIGS, F_SET, MODES, POWER, energies

BITS = (2, 4, 8)


def load(path, p):
    rows = np.load(path)["rows"]
    e = np.min(np.stack([energies(rows, p, f)[1] for f in F_SET]), axis=0)
    return {(int(r[0]), int(r[1]), int(r[2])): float(v) for r, v in zip(rows, e)}


class Problem:
    def __init__(self, biased, acc, thr):
        self.b, self.acc, self.thr = biased, acc, thr
        self.idx = {tuple(c): i for i, c in enumerate(ALL_CONFIGS)}
        self.modes = sorted({k[1] for k in biased})
        self.T = sorted({k[2] for k in biased})
        self.seen = {}   # key -> biased energy (feasible and accurate only)
        self.evals = 0

    def key(self, x):
        return self.idx[tuple(x[:8])], self.modes[x[8]], self.T[x[9]]

    def f(self, x):
        k = self.key(x)
        self.evals += 1
        e = self.b.get(k, math.inf) if self.acc[k[0]] >= self.thr else math.inf
        if e < math.inf:
            self.seen[k] = e
        return e

    def random(self, rng):
        return [rng.choice(BITS) for _ in range(8)] + [rng.randrange(len(self.modes)), rng.randrange(len(self.T))]

    def neighbour(self, x, rng):
        y = list(x)
        d = rng.randrange(10)
        if d < 8:
            i = BITS.index(y[d])
            y[d] = BITS[max(0, min(2, i + rng.choice((-1, 1))))]
        elif d == 8:
            y[8] = rng.choice([m for m in range(len(self.modes)) if m != y[8]] or [y[8]])
        else:
            y[9] = max(0, min(len(self.T) - 1, y[9] + rng.choice((-1, 1))))
        return y


def anneal(pb, rng, budget):
    x, fx = pb.random(rng), math.inf
    while fx == math.inf and pb.evals < budget:   # random restarts until a valid start
        x = pb.random(rng)
        fx = pb.f(x)
    t0, t1 = 0.05, 0.001
    while pb.evals < budget:
        temp = t0 * (t1 / t0) ** (pb.evals / budget)
        y = pb.neighbour(x, rng)
        fy = pb.f(y)
        if fy < math.inf and (fy <= fx or rng.random() < math.exp(-(fy / fx - 1) / temp)):
            x, fx = y, fy


def genetic(pb, rng, budget, size=40, elite=2, pmut=0.15):
    pop = [pb.random(rng) for _ in range(size)]
    fit = [pb.f(x) for x in pop]
    while pb.evals < budget:
        order = sorted(range(size), key=lambda i: fit[i])
        nxt = [pop[i] for i in order[:elite]]
        while len(nxt) < size:
            a, b = (min(rng.sample(range(size), 3), key=lambda i: fit[i]) for _ in range(2))
            child = [pop[a][d] if rng.random() < 0.5 else pop[b][d] for d in range(10)]
            for _ in range(10):
                if rng.random() < pmut:
                    child = pb.neighbour(child, rng)
            nxt.append(child)
        pop = nxt
        fit = [pb.f(x) for x in pop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--biased", required=True)
    ap.add_argument("--acc", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, nargs="+", default=[2000])
    ap.add_argument("--power", default="nominal")
    ap.add_argument("--out")
    args = ap.parse_args()
    p = POWER[args.power]
    truth, biased = load(args.truth, p), load(args.biased, p)
    model = json.load(open(args.acc))
    acc = np.array([model["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    out = {}
    for m in args.margins:
        thr = model["reference_acc"] - m
        t_opt = min(e for k, e in truth.items() if acc[k[0]] >= thr)
        b_opt = min((e, k) for k, e in biased.items() if acc[k[0]] >= thr)[1]
        for budget in args.budget:
            for name, algo in (("anneal", anneal), ("genetic", genetic)):
                runs = []
                for s in range(args.seeds):
                    rng = random.Random(1000 * s + budget)
                    pb = Problem(biased, acc, thr)
                    algo(pb, rng, budget)
                    order = sorted(pb.seen, key=pb.seen.get)
                    fails, chosen = 0, None
                    for k in order:
                        if k in truth:
                            chosen = k
                            break
                        fails += 1
                    runs.append({"pick_fits": bool(order) and order[0] in truth,
                                 "pick_is_exact_opt": bool(order) and order[0] == b_opt,
                                 "failed_builds": fails, "found_fit": chosen is not None,
                                 "regret_pct": (100 * (truth[chosen] / t_opt - 1)) if chosen else None,
                                 "seen": len(pb.seen)})
                rg = [r["regret_pct"] for r in runs if r["regret_pct"] is not None]
                res = {"runs": len(runs), "pick_infeasible_pct": 100 * np.mean([not r["pick_fits"] for r in runs]),
                       "pick_is_exact_opt_pct": 100 * np.mean([r["pick_is_exact_opt"] for r in runs]),
                       "failed_builds_median": float(np.median([r["failed_builds"] for r in runs])),
                       "failed_builds_max": int(max(r["failed_builds"] for r in runs)),
                       "no_fit_found_pct": 100 * np.mean([not r["found_fit"] for r in runs]),
                       "regret_median_pct": float(np.median(rg)) if rg else None}
                out[f"{m:g}|{budget}|{name}"] = res
                print(f"margin {m:g} budget {budget:5d} {name:8s}: pick infeasible {res['pick_infeasible_pct']:5.1f}% "
                      f"(exact optimum {res['pick_is_exact_opt_pct']:5.1f}%), failed builds median "
                      f"{res['failed_builds_median']:4.1f} max {res['failed_builds_max']:3d}, no fit in archive "
                      f"{res['no_fit_found_pct']:4.1f}%, regret median {res['regret_median_pct']}", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
