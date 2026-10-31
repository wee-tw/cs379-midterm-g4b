"""Contract test: board_agent.Board and the vendored FINN driver on PYNQ v2.5 (fake_pynq/).

Runs the agent's own commands in process, the way run_session.py's plans use them, on all five
finn-examples bitstreams: load (v2.5 hwh parser, clocks from the hwh, FINN driver buffers),
clocks (v2.5 Clocks code), probe_slow, probe_throttle, accuracy, and short cont / single /
rate_slow / jit windows with output checks. Run it with the board image's versions (Python 3.6,
numpy 1.13) and with a current Python; the printed results must agree apart from timing.

  <python> run_contract.py [--out results.jsonl]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "fake_pynq"))
sys.path.insert(0, BOARD)

import numpy as np      # noqa: E402
import board_agent as ba   # noqa: E402
import pynq             # noqa: E402

MODELS = ["cnv-w1a1", "cnv-w1a2", "cnv-w2a2", "tfc-w1a1", "kwsmlp-w3a3"]
FMAX = {"kwsmlp-w3a3": 100.0}
TIMING_KEYS = ("t", "t0", "t1", "images", "images_per_s", "late_frames")


def steps():
    bits = os.path.join(BOARD, "bitfiles", "Pynq-Z1")
    npz = os.path.join(BOARD, "data", "cifar10_test_1000.npz")
    out = [{"cmd": "info"}]
    for m in MODELS:
        fmax = FMAX.get(m, 200.0)
        out += [{"cmd": "load", "model": m, "bitfile": os.path.join(bits, m + ".bit"), "batch": 1000},
                {"cmd": "clock", "mhz": 100.0}, {"cmd": "probe_slow"}, {"cmd": "probe_throttle"}]
        if m.startswith("cnv"):
            out.append({"cmd": "accuracy", "npz": npz, "n": 1000})
        out += [{"cmd": "clock", "mhz": fmax},
                {"cmd": "window", "state": "cont", "seconds": 0.3},
                {"cmd": "window", "state": "single", "seconds": 0.3},
                {"cmd": "window", "state": "rate_slow", "rate": 100.0, "run_mhz": fmax, "seconds": 0.3},
                {"cmd": "clock", "mhz": 35.714},
                {"cmd": "window", "state": "jit", "rate": 300.0, "batch": 300, "seconds": 0.3},
                {"cmd": "clock", "mhz": round(1000.0 / 3969, 4)},   # slowest FCLK0 (63 x 63)
                {"cmd": "clock", "mhz": fmax}]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()
    agent = ba.Agent(ba.Board())
    rows = []
    for c in steps():
        try:
            r = getattr(agent, "cmd_" + c["cmd"])(c)
            r.update({"ok": r.get("ok", True), "cmd": c["cmd"]})
        except Exception as e:   # as board_agent.main: report and keep going
            r = {"ok": False, "cmd": c["cmd"], "error": "%s: %s" % (type(e).__name__, e)}
        r = {k: v for k, v in r.items() if k not in TIMING_KEYS + ("tried", "before", "after", "regs")}
        r["step"] = c.get("state") or c["cmd"]
        rows.append(r)
        print(json.dumps(r, sort_keys=True), flush=True)
    summary = {"python": sys.version.split()[0], "numpy": np.__version__, "pynq": pynq.__version__,
               "steps": len(rows), "errors": [r for r in rows if r.get("error")],
               "outputs_mismatch": [r for r in rows if r.get("outputs_match") is False],
               "flushes": pynq.Xlnk.flushes, "invalidates": pynq.Xlnk.invalidates}
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
