"""Add the timing-only plan (no power meter) to run_session.py."""
import sys

p = sys.argv[1]
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    assert old in t, old[:80]
    t = t.replace(old, new)


rep('''  python run_session.py --mock --plan quick                      # test without a board
  python run_session.py --host xilinx@192.168.2.99 --push --plan quick
  python run_session.py --host xilinx@192.168.2.99 --plan h2prime --meter manual''',
    '''  python run_session.py --mock --plan quick                      # test without a board
  python run_session.py --host xilinx@192.168.2.99 --push --plan quick
  python run_session.py --host xilinx@192.168.2.99 --plan h2prime --meter manual
  python run_session.py --host xilinx@192.168.2.99 --plan timing  # no power meter

Plans: quick (setup check), h2prime (power session, needs a meter log or --meter manual),
timing (no meter: throughput and single-image latency at each clock, whether race-to-halt,
the slowest-divider policy and the just-in-time clocks keep up with the rate, and the policy-map
clocks; analysed by analyze_timing.py).''')

rep('''def plan_steps(name, models, clocks, rates, secs, reps, npz, throttle=True):
    """List of (kind, payload). kind 'cmd' sends a command; 'win' is a measured window."""
    steps = [("cmd", {"cmd": "info"})]''',
    '''def timing_steps(models, clocks, rates, npz, throttle, reps):
    """Timing-only session (no power meter): the PL's cycles per image and per single image at
    each clock (cont, single), and whether each fixed-rate policy keeps up (late frames)."""
    steps = [("cmd", {"cmd": "info"})]
    for m in models:
        steps.append(("cmd", {"cmd": "load", "model": m, "bitfile": f"{BITS}/{m}.bit",
                              "batch": 1000}))
        steps.append(("cmd", {"cmd": "clock", "mhz": 100.0}))
        steps.append(("cmd", {"cmd": "probe_slow"}))
        if throttle:
            steps.append(("cmd", {"cmd": "probe_throttle"}))
            steps.append(("cmd", {"cmd": "verify_throttle"}))
        if m.startswith("cnv"):
            steps.append(("cmd", {"cmd": "accuracy", "npz": npz, "n": 1000}))
        fs = [f for f in clocks if f <= MAX_MHZ.get(m, 100.0)]
        for f in fs:
            steps.append(("cmd", {"cmd": "clock", "mhz": f}))
            for _ in range(reps):
                steps.append(("win", {"state": "cont"}))
                steps.append(("win", {"state": "single"}))
        fmax = max(fs)
        steps.append(("cmd", {"cmd": "clock", "mhz": fmax}))
        for R in rates:
            steps.append(("win", {"state": "rate_halt", "rate": R, "run_mhz": fmax}))
            steps.append(("win", {"state": "rate_slow", "rate": R, "run_mhz": fmax}))
        if m.startswith("cnv"):
            for k in (2, 4, 8):
                steps.append(("win", {"state": "rate_halt", "rate": max(rates), "run_mhz": fmax,
                                      "k": k}))
        if m == "cnv-w1a1":   # the policy-map clocks at 300 FPS (predictions.md)
            for k in (1, 2):
                steps.append(("win", {"state": "rate_halt", "rate": 300.0, "run_mhz": fmax, "k": k}))
            for f in (35.714, 22.727, 10.417):
                steps.append(("cmd", {"cmd": "clock", "mhz": f}))
                steps.append(("win", {"state": "jit", "rate": 300.0, "batch": 300}))
        cyc = est_cycles(m)
        for R in (rates if cyc else []):
            f_jit = max(round(1.05 * R * cyc / 1e6, 4), F_MIN)
            if f_jit <= fmax:
                steps.append(("cmd", {"cmd": "clock", "mhz": f_jit}))
                steps.append(("win", {"state": "jit", "rate": R, "batch": max(1, int(R))}))
        steps.append(("cmd", {"cmd": "clock", "mhz": fmax}))
    return steps


def plan_steps(name, models, clocks, rates, secs, reps, npz, throttle=True):
    """List of (kind, payload). kind 'cmd' sends a command; 'win' is a measured window."""
    if name == "timing":
        steps = timing_steps(models, clocks, rates, npz, throttle, reps)
        if not throttle:
            steps = [s for s in steps if not (s[0] == "win" and s[1]["state"].endswith("_halt"))]
        return [(k, dict(p, seconds=secs) if k == "win" else p) for k, p in steps]
    steps = [("cmd", {"cmd": "info"})]''')

rep('''    p.add_argument("--plan", default="quick", choices=["quick", "h2prime"])''',
    '''    p.add_argument("--plan", default="quick", choices=["quick", "h2prime", "timing"])''')
rep('''    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--reps", type=int, default=3)''',
    '''    p.add_argument("--seconds", type=float, default=None, help="window length (12 s; timing: 6 s)")
    p.add_argument("--reps", type=int, default=None, help="repeats per state (3; timing: 2)")''')
rep('''    args = p.parse_args()
    if args.push and not args.mock:''',
    '''    args = p.parse_args()
    if args.seconds is None:
        args.seconds = 6.0 if args.plan == "timing" else 12.0
    if args.reps is None:
        args.reps = 2 if args.plan == "timing" else 3
    if args.push and not args.mock:''')
open(p, "w", encoding="utf-8").write(t)
print("run_session patched")
