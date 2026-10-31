"""Model-based energy-aware search for ResNet-8 on PYNQ-Z2 (estimates, not measurements).

Design space per configuration b = (W1,A1,...,W4,A4) in {2,4,8}^8:
  mode     lut / dsp / lut_finn (see cost_tables.py)
  T        target cycles per frame given to FINN's SetFolding (folding)
  variants per-node LUT/DSP and BRAM/LUTRAM choices (choose_variants)
  f        PL clock (MHz)

Board power model (literature-calibrated, see implementation_notes.md section 6):
  P_idle(f, U)   = P0 + k_f*f + k_fU*f*U        overlay loaded, clocks running, no work
  P_active(f, U) = P_idle(f, U) + k_act*f*U      streaming inference
  U = weighted utilisation (LUT, BRAM18, DSP fractions of the device)
Energy per inference:
  continuous   E_c = P_active / FPS_max,   FPS_max = f*1e6 / T_cyc
  fixed rate R E_r = P_idle / R + k_act*U*T_cyc*1e-6      (requires FPS_max >= R)

Baselines (same model, same constraints):
  B1 tuned uniform   A' restricted to configurations with one (W,A) for all groups
  B2 proxy           bits minimising BOPs under the accuracy constraint, FINN default
                     folding (lut_finn, auto memory) at the fastest feasible T, 100 MHz
  B3 sequential      same bits as B2, then the hardware search maximises throughput
  B4 latency-opt     A' search space, but maximise throughput instead of minimising energy

  python energy_search.py points --tables tables.json --out points.npz
  python energy_search.py search --points points.npz --acc acc_model.json --out result.json
"""

import argparse
import itertools
import json
import time

import numpy as np

from assemble_costs import DEVICE, RES, assemble, choose_variants, load_tables

BITS = (2, 4, 8)
ALL_CONFIGS = [c for c in itertools.product(BITS, repeat=8)]
# later modes are appended so the mode indices stored in older points files keep their
# meaning, and modes missing from a cost table are skipped: dsp4 (DSP-aware refolding),
# dspc / dsp4c (MVAU_rtl DSPs counted as the RTL instantiates them, cost_tables.py)
MODES = ("lut", "dsp", "lut_finn", "dsp4", "dspc", "dsp4c")
# (name, weights, MACs, group, index of the input activation in A1..A4 or None for the image)
LAYERS = [("stem", 432, 442368, 0, None), ("s1c1", 2304, 2359296, 0, 0),
          ("s1c2", 2304, 2359296, 0, 0), ("s2c1", 4608, 1179648, 1, 0),
          ("s2c2", 9216, 2359296, 1, 1), ("s2sc", 512, 131072, 1, 0),
          ("s3c1", 18432, 1179648, 2, 1), ("s3c2", 36864, 2359296, 2, 2),
          ("s3sc", 2048, 131072, 2, 1), ("fc", 640, 640, 3, 3)]
# budget derating for what the estimates leave out (PYNQ shell, FIFOs, width converters)
DERATE = {"LUT": 0.80, "BRAM_18K": 0.70, "DSP": 1.00}
# U for power: LUT fraction plus a smaller weight for BRAM/DSP fractions (assumption;
# the BISMO fit below uses the LUT fraction, BRAM was ~90% in all its instances)
UTIL_W = {"LUT": 1.0, "BRAM_18K": 0.25, "DSP": 0.25}
F_SET = (50.0, 66.67, 100.0, 125.0)
R_SET = (30.0, 300.0, 1000.0, 3000.0, 5000.0)
POWER = {  # W and W/MHz, see implementation_notes.md section 7
    # least-squares fit to BISMO's six idle measurements on PYNQ-Z1 (USB meter, whole
    # board), residuals < 0.05 W; k_act = mean execute increment / (f*U)
    "bismo_fit": dict(P0=1.445, k_f=0.00459, k_fU=0.00290, k_act=0.0052),
    "nominal": dict(P0=1.60, k_f=0.0030, k_fU=0.0015, k_act=0.0035),
    # Gutierrez 2026 (PYNQ-Z1, 12 V input): 2.00 -> 2.18 W from 100 to 200 MHz under load
    "flat": dict(P0=1.80, k_f=0.0010, k_fU=0.0, k_act=0.0013),
    # anchored to the one PYNQ-Z2 measurement of a FINN design we have: CNV-W1A1 in
    # MLPerf Tiny v0.7, 1.67 W while inferring (bare metal). Post-route utilisation of
    # that build (Borras 2022, Table 5: 46.1% LUT, 71.4% BRAM, 0 DSP; U ~ 0.64) gives
    # 1.70 W at 100 MHz; Petrica's 49% / 88% / 11% (U ~ 0.74) gives 1.72 W
    "z2_anchor": dict(P0=1.25, k_f=0.0030, k_fU=0.0010, k_act=0.0013),
}


def bops(b):
    w, a = b[0::2], b[1::2]
    return sum(macs * w[g] * (8 if ai is None else a[ai]) for _, _, macs, g, ai in LAYERS)


def wmem_bits(b):
    w = b[0::2]
    return sum(n * w[g] for _, n, _, g, _ in LAYERS)


def is_uniform(b):
    return len(set(zip(b[0::2], b[1::2]))) == 1


def cmd_points(args):
    table, grid = load_tables(args.tables)
    if args.derate_lut is not None:
        DERATE["LUT"] = args.derate_lut
    if args.derate_bram is not None:
        DERATE["BRAM_18K"] = args.derate_bram
    if args.derate_dsp is not None:   # a safety margin against DSP under-estimation
        DERATE["DSP"] = args.derate_dsp
    budget = {k: DEVICE[k] * DERATE[k] for k in RES}
    print("budget", budget, flush=True)
    rows = []
    t0 = time.time()
    for ci, b in enumerate(ALL_CONFIGS[:args.limit]):
        for mi, mode in enumerate(MODES):
            for T in grid:
                nodes = assemble(table, list(b), mode, T)
                if nodes is None:
                    continue
                cyc = max(n["cycles"] for n in nodes)
                if mode == "lut_finn":  # FINN default: auto memory, no per-node choice
                    tot = {k: sum(n["auto"][k] for n in nodes) for k in RES}
                    if any(tot[k] > budget[k] for k in RES):
                        continue
                else:
                    pick, tot = choose_variants(nodes, budget, UTIL_W)
                    if pick is None:
                        continue
                rows.append((ci, mi, T, cyc, tot["LUT"], tot["BRAM_18K"], tot["DSP"]))
        if ci % 500 == 0:
            print(f"{ci}/{len(ALL_CONFIGS)} configs, {len(rows)} feasible points, "
                  f"{time.time() - t0:.0f} s", flush=True)
    np.savez(args.out, rows=np.array(rows, dtype=np.float64))
    print("saved", args.out, len(rows), "points")


def utilisation(lut, bram, dsp):
    return (UTIL_W["LUT"] * lut / DEVICE["LUT"] + UTIL_W["BRAM_18K"] * bram / DEVICE["BRAM_18K"]
            + UTIL_W["DSP"] * dsp / DEVICE["DSP"])


def energies(rows, p, f):
    """Per design point at clock f: FPS, continuous energy (J), idle power, dynamic J/frame."""
    cyc, U = rows[:, 3], utilisation(rows[:, 4], rows[:, 5], rows[:, 6])
    fps = f * 1e6 / cyc
    p_idle = p["P0"] + p["k_f"] * f + p["k_fU"] * f * U
    p_act = p_idle + p["k_act"] * f * U
    e_dyn = p["k_act"] * U * cyc * 1e-6
    return fps, p_act / fps, p_idle, e_dyn


def describe(rows, i, f):
    ci, mi, T, cyc, lut, bram, dsp = rows[i]
    b = ALL_CONFIGS[int(ci)]
    return {"bits": list(b), "mode": MODES[int(mi)], "T_target": int(T), "cycles": int(cyc),
            "f_MHz": f, "FPS": round(f * 1e6 / cyc, 1), "LUT": int(lut), "BRAM18": float(bram),
            "DSP": int(dsp), "U": round(float(utilisation(lut, bram, dsp)), 3),
            "BOPs_M": round(bops(b) / 1e6, 1), "wmem_kbit": wmem_bits(b) / 1e3}


def search(rows, acc, thr, p):
    """Return the selections of A', B1..B4 for both regimes under power parameters p."""
    cfg = rows[:, 0].astype(int)
    ok_acc = np.array([acc[c] >= thr for c in range(len(ALL_CONFIGS))])
    feas = ok_acc[cfg]
    uni = np.array([is_uniform(b) for b in ALL_CONFIGS])[cfg]
    out = {"continuous": {}, "fixed_rate": {}}
    per_f = {f: energies(rows, p, f) for f in F_SET}

    def best(mask, key):  # key(f) -> array to minimise
        best_v, best_i, best_f = np.inf, None, None
        for f in F_SET:
            v = np.where(mask, key(f), np.inf)
            i = int(np.argmin(v))
            if v[i] < best_v:
                best_v, best_i, best_f = v[i], i, f
        return best_i, best_f, best_v

    U_all = utilisation(rows[:, 4], rows[:, 5], rows[:, 6])

    def fastest(mask):  # max FPS at the highest clock; ties go to the fewest resources (U)
        fps = np.where(mask, per_f[max(F_SET)][0], -np.inf)
        top = np.where(fps >= fps.max() * (1 - 1e-9))[0]
        return int(top[np.argmin(U_all[top])])

    # proxy bits (B2/B3): minimum BOPs among accurate configurations
    acc_cfgs = [c for c in range(len(ALL_CONFIGS)) if ok_acc[c]]
    b2 = min(acc_cfgs, key=lambda c: (bops(ALL_CONFIGS[c]), -acc[c])) if acc_cfgs else None

    regimes = [("continuous", None)] + [("fixed_rate", R) for R in R_SET]
    for regime, R in regimes:
        def energy(f):
            fps, e_c, p_idle, e_dyn = per_f[f]
            if R is None:
                return e_c
            return np.where(fps >= R, p_idle / R + e_dyn, np.inf)

        sel = {}
        i, f, e = best(feas, energy)
        sel["A_prime"] = (i, f)
        i, f, e = best(feas & uni, energy)
        sel["B1_uniform"] = (i, f)
        # B4: fastest design (max FPS at the highest clock), energy reported for it
        fmax = max(F_SET)
        sel["B4_latency"] = (fastest(feas), fmax)
        # B2 runs at FINN's default 100 MHz, or at the cap of a timing class below it
        # (timing_safe.py: a set capped under 100 MHz holds designs that cannot reach it)
        f_b2 = 100.0 if 100.0 in F_SET else fmax
        if b2 is not None:
            m_b2 = (cfg == b2) & (rows[:, 1] == MODES.index("lut_finn"))
            if m_b2.any():
                fps_b2 = np.where(m_b2, per_f[f_b2][0], -np.inf)
                sel["B2_proxy"] = (int(np.argmax(fps_b2)), f_b2)
            m_b3 = (cfg == b2) & (rows[:, 1] != MODES.index("lut_finn"))
            if m_b3.any():
                sel["B3_sequential"] = (fastest(m_b3), fmax)
        res = {}
        for name, (i, f) in sel.items():
            if i is None:
                continue
            fps, e_c, p_idle, e_dyn = (x[i] for x in per_f[f])
            d = describe(rows, i, f)
            d["acc_est"] = round(float(acc[int(rows[i, 0])]), 2)
            d["E_continuous_mJ"] = round(float(e_c) * 1e3, 4)
            # share of the energy that is idle (clock + static) power rather than activity
            p_act = float(e_c * fps)
            d["idle_share_continuous"] = round(float(p_idle) / p_act, 3)
            if R is not None:
                d["E_fixed_mJ"] = (round(float(p_idle / R + e_dyn) * 1e3, 4)
                                   if fps >= R else None)
                if fps >= R:
                    d["idle_share_fixed"] = round(float((p_idle / R) / (p_idle / R + e_dyn)), 3)
            res[name] = d
        if R is None:
            out["continuous"] = res
        else:
            out["fixed_rate"][str(int(R))] = res
    return out


def cmd_search(args):
    rows = np.load(args.points)["rows"]
    if args.acc:
        a = json.load(open(args.acc))
        acc = [a["acc"][",".join(map(str, b))] for b in ALL_CONFIGS]
        thr = a["threshold"]
    else:
        acc, thr = [100.0] * len(ALL_CONFIGS), 0.0
    out = {name: search(rows, acc, thr, p) for name, p in POWER.items()}
    json.dump(out, open(args.out, "w"), indent=1)
    for name in out:
        print("== power params:", name)
        for regime, res in [("continuous", out[name]["continuous"])] + [
                (f"fixed R={R}", r) for R, r in out[name]["fixed_rate"].items()]:
            print("  ", regime)
            for k, d in res.items():
                e = d.get("E_fixed_mJ", d["E_continuous_mJ"]) if "fixed" in regime else d["E_continuous_mJ"]
                print(f"     {k:14s} E={e} mJ  FPS={d['FPS']}  f={d['f_MHz']}  bits={d['bits']} "
                      f"mode={d['mode']} LUT={d['LUT']} BRAM18={d['BRAM18']} DSP={d['DSP']} "
                      f"acc={d['acc_est']}")


F_EXT = (5.0, 10.0, 20.0, 25.0, 33.33, 40.0, 50.0, 66.67, 100.0, 125.0)  # 1000 MHz / divisors
# every FCLK0 a PYNQ-Z2 can generate: IO PLL 1000 MHz / (DIVISOR0 * DIVISOR1), each 1..63
F_JIT = np.unique([1000.0 / (a * b) for a in range(1, 64) for b in range(1, 64)])
F_SET_FULL, F_EXT_FULL = F_SET, F_EXT


def set_fmax(fmax):
    """Cap the clock grids at fmax MHz and add fmax itself (an FCLK0 setting a timing class
    allows, timing_safe.py). F_JIT is already capped at max(F_SET) where it is used."""
    global F_SET, F_EXT
    F_SET = tuple(sorted({f for f in F_SET_FULL if f <= fmax} | {fmax}))
    F_EXT = tuple(sorted({f for f in F_EXT_FULL if f <= fmax} | {fmax}))


def cmd_idle(args):
    """Fixed-rate energy under three idle policies (model only, not measured).

    running  PL clock keeps running between frames (FINN/PYNQ default), clocks F_SET
    lowclk   same, clock set extended down to 5 MHz (closer to just-in-time clocking)
    gated    race-to-halt: the PS clock throttle (UG585 "Clock Throttle") stops FCLK0
             between frames, so idle power is P0 and
             E = P0/R + (k_f + (k_fU + k_act)*U) * T_cyc   for any f with FPS(f) >= R
    Continuous regime: E_c = (P0/f + k_f + (k_fU + k_act)*U) * T_cyc, so the fastest
    design loses to the energy optimum by at most kappa*U_B4 / (P0/f_max + k_f).

    gated_lat (with --latency): one frame at a time, so the clock has to stay on for the
    frame's latency L (FINN critical path, latency_check.py; times --lat-scale), not just
    for T_cyc; if L/f exceeds the frame period, frames overlap and the clock never stops:
             E = P0/R + (k_f + k_fU*U) * f * min(L*s/f, 1/R) + k_act*U*T_cyc
    """
    rows = np.load(args.points)["rows"]
    if args.acc:
        a = json.load(open(args.acc))
        acc = np.array([a["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
        thr = a["threshold"]
    else:
        acc, thr = np.full(len(ALL_CONFIGS), 100.0), 0.0
    cfg = rows[:, 0].astype(int)
    feas = (acc >= thr)[cfg]
    uni = np.array([is_uniform(b) for b in ALL_CONFIGS])[cfg]
    cyc, U = rows[:, 3], utilisation(rows[:, 4], rows[:, 5], rows[:, 6])
    fmax = max(F_SET)

    def running(p, R, fs):
        best = np.full(len(rows), np.inf)
        fbest = np.zeros(len(rows))
        for f in fs:
            e = np.where(f * 1e6 / cyc >= R,
                         (p["P0"] + p["k_f"] * f + p["k_fU"] * f * U) / R
                         + p["k_act"] * U * cyc * 1e-6, np.inf)
            better = e < best
            best, fbest = np.where(better, e, best), np.where(better, f, fbest)
        return best, fbest

    def gated(p, R):
        e = np.where(fmax * 1e6 / cyc >= R,
                     p["P0"] / R + (p["k_f"] + (p["k_fU"] + p["k_act"]) * U) * cyc * 1e-6, np.inf)
        return e, np.full(len(rows), fmax)

    lat = (np.load(args.latency)["latency_cycles"] * args.lat_scale
           if getattr(args, "latency", None) else None)

    def gated_lat(p, R, fs, deadline=False, gate=True):
        """One frame at a time. deadline: the frame must finish before the next one
        arrives (L/f <= 1/R), otherwise frames may overlap in the pipeline. gate=False
        keeps the clock running (only meaningful with the deadline)."""
        best, fbest = np.full(len(rows), np.inf), np.zeros(len(rows))
        for f in fs:
            ok = f * 1e6 / cyc >= R
            if deadline:
                ok &= lat / (f * 1e6) <= 1.0 / R
            on = np.minimum(lat / (f * 1e6), 1.0 / R) if gate else 1.0 / R   # clock-on (s)
            e = np.where(ok, p["P0"] / R + (p["k_f"] + p["k_fU"] * U) * f * on
                         + p["k_act"] * U * cyc * 1e-6, np.inf)
            better = e < best
            best, fbest = np.where(better, e, best), np.where(better, f, fbest)
        return best, fbest

    def jit(p, R, deadline=False):
        """Clock running at the slowest FCLK0 setting that sustains R (and, with the
        deadline, finishes each frame within 1/R): energy grows with f, so this is the
        best running clock for each design."""
        fj = F_JIT[F_JIT <= fmax]
        need = R * cyc / 1e6
        if deadline:
            need = np.maximum(need, R * lat / 1e6)
        k = np.searchsorted(fj, need)        # first available clock >= need
        ok = k < len(fj)
        f = np.where(ok, fj[np.minimum(k, len(fj) - 1)], fmax)
        e = np.where(ok, (p["P0"] + (p["k_f"] + p["k_fU"] * U) * f) / R
                     + p["k_act"] * U * cyc * 1e-6, np.inf)
        return e, f

    def pick(e, f, mask):
        v = np.where(mask, e, np.inf)
        i = int(np.argmin(v))
        if not np.isfinite(v[i]):
            return {"E_mJ": None}   # no design meets the rate (or the deadline)
        return {"E_mJ": round(float(v[i]) * 1e3, 4), "f_MHz": float(f[i]), "cycles": int(cyc[i]),
                "U": round(float(U[i]), 3), "bits": list(ALL_CONFIGS[cfg[i]]),
                "mode": MODES[int(rows[i, 1])]}

    def fastest(mask, key):  # largest key; ties go to the fewest resources (U), as in search()
        k = np.where(mask, key, -np.inf)
        top = np.where(k >= k.max() * (1 - 1e-9) if k.max() > 0 else k == k.max())[0]
        return int(top[np.argmin(U[top])])

    i4 = fastest(feas, fmax * 1e6 / cyc)  # B4: fastest design
    # B2/B3 as in search(): minimum-BOPs bits among accurate configurations; B2 with FINN's
    # default folding at its fastest T and 100 MHz, B3 with our folding at the fastest T
    fixed = {"B4_latency": (i4, fmax)}
    if lat is not None:   # lowest single-frame latency (FINN critical path) at f_max
        fixed["B4L_min_latency"] = (fastest(feas, -lat.astype(float)), fmax)
    acc_cfgs = [c for c in range(len(ALL_CONFIGS)) if acc[c] >= thr]
    if acc_cfgs:
        b2 = min(acc_cfgs, key=lambda c: (bops(ALL_CONFIGS[c]), -acc[c]))
        lf = rows[:, 1] == MODES.index("lut_finn")
        for name, mask, f in [("B2_proxy", (cfg == b2) & lf, 100.0),
                              ("B3_sequential", (cfg == b2) & ~lf, fmax)]:
            if mask.any():
                fixed[name] = (fastest(mask, f * 1e6 / cyc), f)
    out = {}
    for pname, p in POWER.items():
        kappa = p["k_fU"] + p["k_act"]
        e_c = np.where(feas, (p["P0"] / fmax + p["k_f"] + kappa * U) * cyc * 1e-6, np.inf)
        res = {"continuous": {
            "gap_B4_vs_opt_pct": round(100 * (e_c[i4] / e_c.min() - 1), 2),
            "bound_pct": round(100 * kappa * U[i4] / (p["P0"] / fmax + p["k_f"]), 1)}}
        for R in R_SET:
            pol = {"running": running(p, R, F_SET), "lowclk": running(p, R, F_EXT),
                   "gated": gated(p, R)}
            lat_pol = {}
            if lat is not None:
                lat_pol = {"gated_lat": dict(deadline=False, gate=True),       # streaming
                           "dl_running": dict(deadline=True, gate=False),     # per-frame
                           "dl_gated": dict(deadline=True, gate=True)}        # deadline
                for name, kw in lat_pol.items():
                    pol[name] = gated_lat(p, R, F_EXT, **kw)
                pol["dl_jit"] = jit(p, R, deadline=True)
            pol["jit"] = jit(p, R)
            r = {}
            for name, (e, f) in pol.items():
                r[name] = {"A_prime": pick(e, f, feas), "B1_uniform": pick(e, f, feas & uni)}
                # fixed designs: own clock while running, any clock (f-independent) if gated,
                # own clock (one frame at a time) in the latency-aware policies; B4 with
                # halts is B5, B4L (lowest latency) is the race-to-halt design for deadlines
                for bname, (i, fb) in fixed.items():
                    if name in ("gated", "jit", "dl_jit"):   # f chosen by the policy
                        v = e[i]
                    elif name in lat_pol:
                        v = gated_lat(p, R, (fb,), **lat_pol[name])[0][i]
                    else:
                        v = running(p, R, (fb,))[0][i]
                    r[name][bname + "_mJ"] = (round(float(v) * 1e3, 4) if np.isfinite(v)
                                              else None)
            res[str(int(R))] = r
        out[pname] = res
    json.dump(out, open(args.out, "w"), indent=1)
    for pname, res in out.items():
        c = res["continuous"]
        print(f"== {pname}: continuous B4 gap {c['gap_B4_vs_opt_pct']}% (bound {c['bound_pct']}%)")
        for R in R_SET:
            r = res[str(int(R))]
            for name in [k for k in ("running", "lowclk", "jit", "gated", "gated_lat",
                                     "dl_running", "dl_jit", "dl_gated") if k in r]:
                a, b1 = r[name]["A_prime"], r[name]["B1_uniform"]
                if a["E_mJ"] is None:
                    print(f"   R={int(R):5d} {name:10s} infeasible")
                    continue
                print(f"   R={int(R):5d} {name:10s} A'={a['E_mJ']:9.4f} (f={a['f_MHz']:6.2f} "
                      f"cyc={a['cycles']:7d} U={a['U']:.3f} {a['mode']:8s} {a['bits']})  "
                      f"B1={b1['E_mJ']}  B4={r[name]['B4_latency_mJ']}  "
                      f"B4L={r[name].get('B4L_min_latency_mJ')}  "
                      f"B2={r[name].get('B2_proxy_mJ')}  B3={r[name].get('B3_sequential_mJ')}")


def cmd_pareto(args):
    """Min energy of A' (any configuration) vs B1 (uniform only) at each accuracy floor."""
    rows = np.load(args.points)["rows"]
    a = json.load(open(args.acc))
    acc = np.array([a["acc"][",".join(map(str, b))] for b in ALL_CONFIGS])
    uni = np.array([is_uniform(b) for b in ALL_CONFIGS])
    cfg = rows[:, 0].astype(int)
    floors = np.arange(args.lo, args.hi + 1e-9, args.step)
    out = {}
    for pname, p in POWER.items():
        best = {}
        for regime, R in [("continuous", None)] + [(f"fixed_{int(r)}", r) for r in R_SET]:
            e_cfg = np.full(len(ALL_CONFIGS), np.inf)
            for f in F_SET:
                fps, e_c, p_idle, e_dyn = energies(rows, p, f)
                v = e_c if R is None else np.where(fps >= R, p_idle / R + e_dyn, np.inf)
                np.minimum.at(e_cfg, cfg, v)
            tab = []
            for fl in floors:
                ok = acc >= fl
                ea = e_cfg[ok].min() if ok.any() else np.inf
                eb = e_cfg[ok & uni].min() if (ok & uni).any() else np.inf
                ia = int(np.flatnonzero(ok)[np.argmin(e_cfg[ok])]) if np.isfinite(ea) else None
                tab.append({"acc_floor": round(float(fl), 2),
                            "A_prime_mJ": None if not np.isfinite(ea) else round(ea * 1e3, 4),
                            "B1_mJ": None if not np.isfinite(eb) else round(eb * 1e3, 4),
                            "saving_pct": (None if not (np.isfinite(ea) and np.isfinite(eb))
                                           else round(100 * (1 - ea / eb), 1)),
                            "A_prime_bits": None if ia is None else list(ALL_CONFIGS[ia])})
            best[regime] = tab
        out[pname] = best
    json.dump(out, open(args.out, "w"), indent=1)
    for pname in out:
        print("==", pname)
        for regime, tab in out[pname].items():
            print("  ", regime)
            for t in tab:
                print(f"     acc>={t['acc_floor']:5.1f}  A'={t['A_prime_mJ']}  B1={t['B1_mJ']}  "
                      f"saving={t['saving_pct']}%  bits={t['A_prime_bits']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["points", "search", "pareto", "idle"])
    p.add_argument("--lo", type=float, default=70.0)
    p.add_argument("--hi", type=float, default=86.0)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--derate-lut", type=float, default=None)
    p.add_argument("--derate-bram", type=float, default=None)
    p.add_argument("--derate-dsp", type=float, default=None)
    p.add_argument("--fset", type=float, nargs="+", default=None,
                   help="override the clock set (MHz), e.g. 50 66.67 100 125 142.86")
    p.add_argument("--tables")
    p.add_argument("--limit", type=int, default=None, help="points: only the first N configs (test)")
    p.add_argument("--latency", help="idle: latency.npz from latency_check.py points "
                   "(adds the one-frame-at-a-time gated policy 'gated_lat')")
    p.add_argument("--lat-scale", type=float, default=1.0,
                   help="idle: factor on FINN's critical path (1/1.66 = CNV calibration)")
    p.add_argument("--points")
    p.add_argument("--acc", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--power-json", help="extra parameter sets, e.g. z2_measured from "
                   "board/analyze_session.py: {name: {P0, k_f, k_fU, k_act}}")
    args = p.parse_args()
    if args.power_json:
        POWER.update(json.load(open(args.power_json)))
    if args.fset:
        F_SET = tuple(args.fset)
    {"points": cmd_points, "search": cmd_search, "pareto": cmd_pareto,
     "idle": cmd_idle}[args.cmd](args)
