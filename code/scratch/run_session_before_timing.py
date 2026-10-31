"""Drive a PYNQ-Z2 measurement session from the PC (board side: board_agent.py).

  python run_session.py --mock --plan quick                      # test without a board
  python run_session.py --host xilinx@192.168.2.99 --push --plan quick
  python run_session.py --host xilinx@192.168.2.99 --plan h2prime --meter manual

Needs key-based ssh login and passwordless sudo on the board (set up once by the user;
see README_board.md). Every window is recorded with the PC timestamps around the command
(pc_t0/pc_t1) and the board's own t0/t1, so a power-meter log kept on the PC clock, or
readings typed in with --meter manual, can be matched by analyze_session.py.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/home/xilinx/a_prime"
BITS = REMOTE_DIR + "/bitfiles/Pynq-Z1"
KEY = os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519_pynq")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
FIELDS = ["step", "model", "state", "f_set", "fclk0_mhz", "run_mhz", "rate", "seconds",
          "pc_t0", "pc_t1", "t0", "t1", "images", "images_per_s", "late_frames",
          "temp_c_start", "temp_c_end", "outputs_match", "halts", "halt_fail", "halt_verified", "cpu", "k",
          "power_w", "ok", "error"]


F_MIN = round(1000.0 / (63 * 63), 6)   # slowest FCLK0 the PS dividers can make (MHz)
# design clock of each finn-examples Pynq-Z1 build (hwh PCW_FPGA0_PERIPHERAL_FREQMHZ): never exceed
MAX_MHZ = {"cnv-w1a1": 200.0, "cnv-w1a2": 200.0, "cnv-w2a2": 200.0, "tfc-w1a1": 200.0,
           "tfc-w1a2": 200.0, "tfc-w2a2": 200.0, "kwsmlp-w3a3": 100.0,
           "unsw_nb15-mlp-w2a2": 100.0}


def est_cycles(model):
    """FINN v0.10.1 estimate of the bottleneck cycles per frame (board/est), or None."""
    f = os.path.join(HERE, "est", model, "calibration.json")
    return json.load(open(f))["estimate_finn_v0101"]["max_cycles"] if os.path.exists(f) else None


def plan_steps(name, models, clocks, rates, secs, reps, npz, throttle=True):
    """List of (kind, payload). kind 'cmd' sends a command; 'win' is a measured window."""
    steps = [("cmd", {"cmd": "info"})]
    if name == "quick":
        models, clocks, rates, reps = models[:1], [100.0, 200.0], [100.0], 1
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
        steps.append(("win", {"state": "idle_halt"}))       # clock-independent states first
        steps.append(("win", {"state": "idle_slow"}))
        fs = [f for f in clocks if f <= MAX_MHZ.get(m, 100.0)]
        for f in fs:
            steps.append(("cmd", {"cmd": "clock", "mhz": f}))
            for _ in range(reps):
                steps.append(("win", {"state": "idle"}))
                steps.append(("win", {"state": "cont"}))
                steps.append(("win", {"state": "single"}))
            for R in rates:
                steps.append(("win", {"state": "rate", "rate": R}))
        fmax = max(fs)
        steps.append(("cmd", {"cmd": "clock", "mhz": fmax}))
        for R in rates:
            for _ in range(reps):
                steps.append(("win", {"state": "rate_halt", "rate": R, "run_mhz": fmax}))
                steps.append(("win", {"state": "rate_slow", "rate": R, "run_mhz": fmax}))
                steps.append(("win", {"state": "rate", "rate": R}))
            # the FINN driver's busy wait instead of sleeping: size of the CPU overhead
            steps.append(("win", {"state": "rate", "rate": R, "cpu": "poll"}))
        # batched race-to-halt (policy_map.py): k frames per clock start spread the pipeline
        # fill over k frames, so energy per frame should fall as a + b / k. At the highest
        # rate the fill is a visible share of the frame energy; CNV has the deepest pipeline
        if m.startswith("cnv"):
            for k in (2, 4, 8):
                for _ in range(reps):
                    steps.append(("win", {"state": "rate_halt", "rate": max(rates),
                                          "run_mhz": fmax, "k": k}))
        # policy-map crossover (policy_map.py; predictions in predictions/predictions.md):
        # CNV-W1A1 at 300 FPS, where a 1.3-period deadline should favour batched halt (k = 2)
        # over streaming at the 35.7 MHz deadline clock, a 2-period deadline should favour
        # streaming at 22.7 MHz over k = 2, and jit (10.4 MHz) is the floor
        if m == "cnv-w1a1" and name == "h2prime":
            R_x = 300.0
            steps.append(("cmd", {"cmd": "clock", "mhz": fmax}))
            for k in (1, 2):
                for _ in range(reps):
                    steps.append(("win", {"state": "rate_halt", "rate": R_x, "run_mhz": fmax,
                                          "k": k}))
            for f in (35.714, 22.727, 10.417):
                steps.append(("cmd", {"cmd": "clock", "mhz": f}))
                for _ in range(reps):
                    steps.append(("win", {"state": "jit", "rate": R_x, "batch": int(R_x)}))
        # streaming at a just-in-time clock (5% above R x FINN's bottleneck cycles): the
        # latency-aware model predicts this beats one-frame-at-a-time race-to-halt
        cyc = est_cycles(m)
        for R in (rates if cyc else []):
            # never below the slowest FCLK0 (1000 MHz / 63 / 63); the agent then paces the
            # batches to R, i.e. the lowest clock with idle gaps (TFC, KWS at 100 FPS)
            f_jit = max(round(1.05 * R * cyc / 1e6, 4), F_MIN)
            if f_jit <= fmax:
                steps.append(("cmd", {"cmd": "clock", "mhz": f_jit}))
                for _ in range(reps):
                    steps.append(("win", {"state": "jit", "rate": R, "batch": max(1, int(R))}))
        steps.append(("cmd", {"cmd": "clock", "mhz": fmax}))
        for _ in range(reps):
            steps.append(("win", {"state": "idle_halt"}))
            steps.append(("win", {"state": "idle_slow"}))
            steps.append(("win", {"state": "idle"}))
    if not throttle:   # slowest-divider policy only (clock throttle unusable or not wanted)
        steps = [s for s in steps if not (s[0] == "win" and s[1]["state"].endswith("_halt"))]
    return [(k, dict(p, seconds=secs) if k == "win" else p) for k, p in steps]


class Agent:
    def __init__(self, args):
        if args.mock:
            cmd = [sys.executable, "-u", os.path.join(HERE, "board_agent.py"), "--mock"]
        else:
            remote = ("sudo -n bash -lc 'source /etc/profile.d/pynq_venv.sh; "
                      "source /etc/profile.d/xrt_setup.sh; "
                      f"cd {REMOTE_DIR} && python3 -u board_agent.py'")
            cmd = ["ssh", "-i", args.key] + SSH_OPTS + [args.host, remote]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=sys.stderr, text=True, bufsize=1)
        first = json.loads(self.p.stdout.readline())
        assert first.get("ready"), first

    def send(self, c):
        self.p.stdin.write(json.dumps(c) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("agent exited")
        return json.loads(line)

    def close(self):
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        self.p.wait(timeout=30)


def push(args):
    """Copy the agent, its driver files, the CIFAR-10 test subset and the bitfiles (scp)."""
    ssh = ["ssh", "-i", args.key] + SSH_OPTS + [args.host]
    scp = ["scp", "-i", args.key] + SSH_OPTS + ["-r"]
    subprocess.run(ssh + [f"mkdir -p {BITS}"], check=True)
    files = [os.path.join(HERE, "board_agent.py"), os.path.join(HERE, "finn_driver"),
             args.test_npz]
    subprocess.run(scp + files + [f"{args.host}:{REMOTE_DIR}/"], check=True)
    bits = [os.path.join(args.bitdir, f"{m}.{e}") for m in args.models for e in ("bit", "hwh")]
    subprocess.run(scp + bits + [f"{args.host}:{BITS}/"], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="xilinx@192.168.2.99")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--key", default=KEY, help="ssh private key (default ~/.ssh/id_ed25519_pynq)")
    p.add_argument("--push", action="store_true", help="copy agent/driver/bitfiles/data first")
    p.add_argument("--bitdir", default=os.path.join(HERE, "bitfiles", "Pynq-Z1"))
    p.add_argument("--test-npz", default=os.path.join(HERE, "data", "cifar10_test_1000.npz"))
    p.add_argument("--plan", default="quick", choices=["quick", "h2prime"])
    p.add_argument("--models", nargs="+",
                   default=["cnv-w1a1", "cnv-w1a2", "cnv-w2a2", "tfc-w1a1", "kwsmlp-w3a3"],
                   help="TFC/KWS widen the utilisation range for the k_f / k_fU fit")
    p.add_argument("--clocks", type=float, nargs="+", default=[25.0, 50.0, 100.0, 200.0])
    p.add_argument("--rates", type=float, nargs="+", default=[100.0, 1000.0])
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--no-throttle", action="store_true",
                   help="skip the clock-throttle probe and the *_halt windows")
    p.add_argument("--meter", default="none", choices=["none", "manual"],
                   help="manual: type the meter's average power after each window")
    p.add_argument("--out", default=os.path.join(HERE, "sessions"))
    args = p.parse_args()
    if args.push and not args.mock:
        push(args)
    npz = args.test_npz if args.mock else REMOTE_DIR + "/" + os.path.basename(args.test_npz)
    steps = plan_steps(args.plan, args.models, args.clocks, args.rates, args.seconds,
                       args.reps, npz, throttle=not args.no_throttle)
    sdir = os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S") + ("_mock" if args.mock else ""))
    os.makedirs(sdir, exist_ok=True)
    json.dump({"args": vars(args), "steps": steps}, open(os.path.join(sdir, "plan.json"), "w"),
              indent=1)
    log = open(os.path.join(sdir, "responses.jsonl"), "w")
    wf = open(os.path.join(sdir, "windows.csv"), "w", newline="")
    w = csv.DictWriter(wf, fieldnames=FIELDS)
    w.writeheader()
    agent = Agent(args)
    model = None
    est = sum(s[1]["seconds"] for s in steps if s[0] == "win")
    print(f"session {sdir}: {len(steps)} steps, about {est / 60:.0f} min of windows", flush=True)
    for i, (kind, payload) in enumerate(steps):
        if kind == "cmd":
            pc0 = time.time()
            r = agent.send(payload)
            pc1 = time.time()
            r.update({"step": i, "pc_t0": pc0, "pc_t1": pc1})
            log.write(json.dumps(r) + "\n")
            log.flush()
            if payload["cmd"] == "load":
                model = payload["model"]
            print(f"[{i}] {payload['cmd']}: " + json.dumps(
                {k: v for k, v in r.items() if k not in ("tried", "before", "after")})[:300],
                flush=True)
            continue
        c = dict(payload, cmd="window")
        pc0 = time.time()
        r = agent.send(c)
        pc1 = time.time()
        r.update({"step": i, "pc_t0": pc0, "pc_t1": pc1})
        log.write(json.dumps(r) + "\n")
        log.flush()
        row = {k: r.get(k) for k in FIELDS}
        row.update({"model": model, "state": payload["state"], "f_set": payload.get("run_mhz"),
                    "seconds": payload["seconds"], "rate": payload.get("rate")})
        if args.meter == "manual" and r.get("ok"):
            v = input(f"  window {i} {payload['state']}: average power on the meter (W)? ")
            row["power_w"] = float(v) if v.strip() else None
        w.writerow(row)
        wf.flush()
        print(f"[{i}] {payload['state']:9s} ok={r.get('ok')} f={r.get('fclk0_mhz')} "
              f"img/s={r.get('images_per_s')} late={r.get('late_frames')} "
              f"match={r.get('outputs_match')} {r.get('error', '')}", flush=True)
    agent.close()
    print("done:", sdir)


if __name__ == "__main__":
    main()
