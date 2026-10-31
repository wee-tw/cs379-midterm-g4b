"""Measurement agent for PYNQ-Z2 (runs ON THE BOARD as root, inside the PYNQ venv).

Reads one JSON command per line on stdin and answers one JSON line on stdout, so the PC
can drive a whole session over ssh (code/board/run_session.py). Board timestamps are
time.time(); the PC keeps its own. With --mock the accelerator, PYNQ clocks and the SLCR
are simulated, so the protocol can be tested on the PC.

Commands ("cmd" plus fields):
  info                                  board/clock/temperature summary
  load        model, bitfile, batch     load a finn-examples overlay (Pynq-Z1 bitfiles run on
                                        the Z2: same xc7z020clg400 part, no board I/O used)
  clock       mhz                       set FCLK0 through PYNQ Clocks, returns the actual value
  window      state, seconds, rate?, run_mhz?
     idle        overlay loaded, clock running, no work
     idle_halt   FCLK0 stopped by the PS clock throttle for the whole window
     idle_slow   FCLK0 at the slowest divider (~0.25 MHz) for the whole window
     cont        back-to-back batches at the current clock
     single      back-to-back one-image runs (latency-bound, MLPerf single-stream style)
     rate        one frame every 1/rate s, clock running between frames
     rate_halt   race-to-halt: clock started for each frame, stopped after it (B5)
     rate_slow   clock at run_mhz for each frame, slowest divider between frames
  probe_throttle                        find a stop/restart sequence; checks outputs after
  probe_slow                            slowest divider and back; checks outputs after
  accuracy    npz, n                    top-1 on test images (x_test N,32,32,3 uint8; y_test)
  quit

Safety: the AXI-lite control path is clocked by FCLK0, so the agent never touches PL
registers while the clock is stopped. If a restart cannot be confirmed in the throttle
status register, PL commands are refused until the overlay is reloaded (or the board is
power-cycled).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SLCR_BASE = 0xF8000000
SLCR_UNLOCK, UNLOCK_KEY = 0x008, 0xDF0D
FPGA0_CLK_CTRL = 0x170
FPGA0_THR_CTRL, FPGA0_THR_CNT, FPGA0_THR_STA = 0x174, 0x178, 0x17C   # UG585 "Clock Throttle"
STA_RUNNING = 1 << 16          # reads 0x10000 while free running (observed by AMD forum users)
# candidate sequences, (register, value) writes; tried in order by probe_throttle
STOP_SEQS = {"A": [(FPGA0_THR_CNT, 1), (FPGA0_THR_CTRL, 1)],
             "B": [(FPGA0_THR_CTRL, 2), (FPGA0_THR_CTRL, 0), (FPGA0_THR_CNT, 1), (FPGA0_THR_CTRL, 1)],
             "C": [(FPGA0_THR_CNT, 1), (FPGA0_THR_CTRL, 3)]}
START_SEQS = {"A": [(FPGA0_THR_CNT, 0), (FPGA0_THR_CTRL, 2), (FPGA0_THR_CTRL, 0), (FPGA0_THR_CTRL, 1)],
              "B": [(FPGA0_THR_CNT, 0), (FPGA0_THR_CTRL, 1)],
              "C": [(FPGA0_THR_CNT, 0), (FPGA0_THR_CTRL, 0)]}
MODEL_FUNCS = {"cnv-w1a1": "cnv_w1a1_cifar10", "cnv-w1a2": "cnv_w1a2_cifar10",
               "cnv-w2a2": "cnv_w2a2_cifar10", "tfc-w1a1": "tfc_w1a1_mnist",
               "tfc-w1a2": "tfc_w1a2_mnist", "tfc-w2a2": "tfc_w2a2_mnist",
               "kwsmlp-w3a3": "kws_mlp", "unsw_nb15-mlp-w2a2": "mlp_w2a2_unsw_nb15"}


def die_temp_c():
    for d in glob.glob("/sys/bus/iio/devices/iio:device*"):
        try:
            raw = float(open(os.path.join(d, "in_temp0_raw")).read())
            off = float(open(os.path.join(d, "in_temp0_offset")).read())
            scale = float(open(os.path.join(d, "in_temp0_scale")).read())
            return round((raw + off) * scale / 1000, 2)
        except OSError:
            continue
    return None


class Board:
    """Real PYNQ access: SLCR registers, PL clock, finn-examples overlay."""

    def __init__(self):
        from pynq import MMIO
        from pynq.ps import Clocks
        self.Clocks = Clocks
        self.slcr = MMIO(SLCR_BASE, 0x200)
        self.slcr.write(SLCR_UNLOCK, UNLOCK_KEY)   # Linux normally leaves the SLCR unlocked
        self.accel = None

    def rd(self, off):
        return int(self.slcr.read(off))

    def wr(self, off, val):
        self.slcr.write(off, val)

    def get_mhz(self):
        return float(self.Clocks.fclk0_mhz)

    def set_mhz(self, mhz):
        self.Clocks.fclk0_mhz = mhz
        return self.get_mhz()

    def set_slowest(self):
        self.Clocks.set_pl_clk(0, div0=63, div1=63)
        return self.get_mhz()

    def load(self, model, bitfile, batch):
        drv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finn_driver")
        if os.path.isdir(drv):        # vendored FINN v0.10.1 driver files, no pip needed
            sys.path.insert(0, drv)
            from driver_base import FINNExampleOverlay
            from io_shapes import IO_SHAPES
            self.accel = FINNExampleOverlay(bitfile, "zynq-iodma", IO_SHAPES[model],
                                            batch_size=batch, fclk_mhz=100.0)
        else:                         # pip-installed finn-examples
            from finn_examples import models
            self.accel = getattr(models, MODEL_FUNCS[model])(bitfile_path=bitfile)
            self.accel.batch_size = batch

    def run(self, batch):
        self.accel.execute_on_buffers(batch_size=batch)   # blocks until the output DMA is done

    def set_input(self, x):
        a = self.accel
        a.copy_input_data_to_device(a.pack_input(a.fold_input(x)))

    def output(self):
        a = self.accel
        a.copy_output_data_from_device(a.obuf_packed[0])
        return a.unfold_output(a.unpack_output(a.obuf_packed[0])).copy()

    def ishape(self):
        return self.accel.ishape_normal()

    def idt(self):
        return self.accel.idt()


class MockBoard(Board):
    """Simulated board for testing the protocol on a PC (no pynq)."""

    def __init__(self):
        self.regs = {FPGA0_THR_CTRL: 0, FPGA0_THR_CNT: 0, FPGA0_THR_STA: STA_RUNNING}
        self.mhz, self.accel, self.batch = 100.0, None, 1

    def rd(self, off):
        return self.regs.get(off, 0)

    def wr(self, off, val):
        self.regs[off] = val
        if off == FPGA0_THR_CTRL and val & 1:   # pretend sequence A works
            stopped = self.regs[FPGA0_THR_CNT] > 0
            self.regs[FPGA0_THR_STA] = 1 if stopped else STA_RUNNING

    def get_mhz(self):
        return self.mhz

    def set_mhz(self, mhz):
        div = max(1, min(63 * 63, round(1000.0 / mhz)))
        self.mhz = round(1000.0 / div, 6)
        return self.mhz

    def set_slowest(self):
        self.mhz = round(1000.0 / 3969, 6)
        return self.mhz

    def load(self, model, bitfile, batch):
        self.accel, self.batch, self.x = model, batch, None
        self.cycles = 16384 if "cnv" in model else 2048

    def run(self, batch):
        assert self.rd(FPGA0_THR_STA) & STA_RUNNING, "PL access while FCLK0 is stopped"
        time.sleep(batch * self.cycles / (self.mhz * 1e6) + 1e-4)

    def set_input(self, x):
        self.x = x

    def output(self):
        n = self.x.shape[0] if self.x is not None else self.batch
        return np.asarray([[int(v) % 10] for v in
                           (self.x.reshape(n, -1).sum(1) if self.x is not None else range(n))])

    def ishape(self):
        return (self.batch, 32, 32, 3)

    def idt(self):
        return None


def make_input(board, seed=0):
    """Fixed random input in the accelerator's input range (UINT8 images, INT8 for KWS)."""
    rng = np.random.default_rng(seed)
    dt = board.idt()
    lo, hi = (0, 255) if dt is None else (int(dt.min()), int(dt.max()))
    t = np.int8 if lo < 0 and hi <= 127 else np.uint8 if hi <= 255 else np.int32
    return rng.integers(lo, hi + 1, size=board.ishape()).astype(t)


class Agent:
    def __init__(self, board):
        self.b = board
        self.stop_seq = self.start_seq = None
        self.stuck = False
        self.golden = None
        self.batch = 1

    # ---- clock throttle -------------------------------------------------------------
    def running(self):
        return bool(self.b.rd(FPGA0_THR_STA) & STA_RUNNING)

    def apply(self, seq):
        for off, val in seq:
            self.b.wr(off, val)

    def halt(self):
        self.apply(STOP_SEQS[self.stop_seq])

    def resume(self):
        self.apply(START_SEQS[self.start_seq])
        if not self.running():
            for name, seq in START_SEQS.items():
                self.apply(seq)
                if self.running():
                    self.start_seq = name
                    return
            self.stuck = True
            raise RuntimeError("FCLK0 did not restart; PL access blocked (reload/power-cycle)")

    def regs(self):
        return {k: hex(self.b.rd(v)) for k, v in
                [("THR_CTRL", FPGA0_THR_CTRL), ("THR_CNT", FPGA0_THR_CNT),
                 ("THR_STA", FPGA0_THR_STA), ("CLK_CTRL", FPGA0_CLK_CTRL)]}

    def check_outputs(self):
        self.b.run(self.batch)
        out = self.b.output()
        return bool(np.array_equal(out, self.golden))

    # ---- commands -------------------------------------------------------------------
    def cmd_info(self, c):
        return {"t": time.time(), "fclk0_mhz": self.b.get_mhz(), "temp_c": die_temp_c(),
                "regs": self.regs(),
                "stop_seq": self.stop_seq, "start_seq": self.start_seq, "stuck": self.stuck}

    def cmd_load(self, c):
        self.batch = int(c.get("batch", 1000))
        self.b.load(c["model"], c.get("bitfile"), self.batch)
        self.stuck = False
        self.b.set_input(make_input(self.b))
        self.b.run(self.batch)
        self.golden = self.b.output()
        return {"model": c["model"], "batch": self.batch, "fclk0_mhz": self.b.get_mhz(),
                "golden_head": self.golden.reshape(-1)[:8].tolist()}

    def cmd_clock(self, c):
        return {"set": c["mhz"], "actual": self.b.set_mhz(float(c["mhz"]))}

    def cmd_probe_throttle(self, c):
        before = self.regs()
        tried = []
        for sname, sseq in STOP_SEQS.items():
            self.apply(sseq)
            time.sleep(0.05)
            stopped = not self.running()
            sta = hex(self.b.rd(FPGA0_THR_STA))
            restarted = None
            for rname, rseq in START_SEQS.items():
                self.apply(rseq)
                time.sleep(0.05)
                if self.running():
                    restarted = rname
                    break
            tried.append({"stop": sname, "stopped": stopped, "sta_after_stop": sta,
                          "restart": restarted})
            if restarted is None:
                self.stuck = True
                return {"ok": False, "tried": tried, "before": before, "stuck": True,
                        "note": "clock did not restart: no PL access; reload or power-cycle"}
            if stopped:
                self.stop_seq, self.start_seq = sname, restarted
                break
        if not self.stop_seq:
            # the status never showed a stop: either nothing stopped or the bit means something
            # else; LAST_CNT = 0 (documented continuous mode) was written by every restart
            # sequence. Do not claim a usable throttle and do not touch the PL here.
            return {"ok": False, "stop_seq": None, "tried": tried, "before": before,
                    "after": self.regs(),
                    "note": "throttle status never changed; use the slowest-divider policy"}
        out_ok = self.check_outputs()   # safe: a stop and a restart were both observed
        return {"ok": out_ok, "stop_seq": self.stop_seq, "start_seq": self.start_seq,
                "outputs_match": out_ok, "tried": tried, "before": before, "after": self.regs()}

    def cmd_probe_slow(self, c):
        f_run = self.b.get_mhz()
        f_low = self.b.set_slowest()
        time.sleep(0.2)
        self.b.set_mhz(f_run)
        out_ok = self.check_outputs()
        return {"ok": out_ok, "slowest_mhz": f_low, "restored_mhz": self.b.get_mhz(),
                "outputs_match": out_ok}

    def cmd_window(self, c):
        state, secs = c["state"], float(c["seconds"])
        if self.stuck:
            raise RuntimeError("PL access blocked (clock restart failed earlier)")
        if state in ("idle_halt", "rate_halt") and not self.stop_seq:
            raise RuntimeError("no working throttle sequence; run probe_throttle first")
        f0 = self.b.get_mhz()
        f_run = float(c.get("run_mhz", f0))
        rate = float(c.get("rate", 0))
        t_temp0 = die_temp_c()
        n = 0
        late = 0
        t0 = time.time()
        end = t0 + secs
        if state == "idle":
            time.sleep(max(0.0, end - time.time()))
        elif state == "idle_halt":
            self.halt()
            time.sleep(max(0.0, end - time.time()))
            self.resume()
        elif state == "idle_slow":
            self.b.set_slowest()
            time.sleep(max(0.0, end - time.time()))
            self.b.set_mhz(f0)
        elif state == "cont":
            while time.time() < end:
                self.b.run(self.batch)
                n += self.batch
        elif state == "single":          # back-to-back batch-1 runs (MLPerf single-stream)
            while time.time() < end:
                self.b.run(1)
                n += 1
        elif state in ("rate", "rate_halt", "rate_slow"):
            period, nxt = 1.0 / rate, time.time()
            if state == "rate_halt":
                self.b.set_mhz(f_run)
                self.halt()
            elif state == "rate_slow":
                self.b.set_slowest()
            while nxt < end and time.time() < end:   # stop on time even if frames are late
                if state == "rate_halt":
                    self.resume()
                elif state == "rate_slow":
                    self.b.set_mhz(f_run)
                self.b.run(1)
                if state == "rate_halt":
                    self.halt()
                elif state == "rate_slow":
                    self.b.set_slowest()
                n += 1
                nxt += period
                slack = nxt - time.time()
                if slack > 0:
                    time.sleep(slack)
                else:
                    late += 1
            if state == "rate_halt":
                self.resume()
            self.b.set_mhz(f0)
        else:
            raise ValueError("unknown state " + state)
        t1 = time.time()
        res = {"state": state, "t0": round(t0, 4), "t1": round(t1, 4), "images": n,
               "images_per_s": round(n / (t1 - t0), 2), "late_frames": late,
               "fclk0_mhz": self.b.get_mhz(), "run_mhz": f_run if "rate" in state else f0,
               "rate": rate or None, "temp_c_start": t_temp0, "temp_c_end": die_temp_c()}
        if state in ("idle_halt", "rate_halt", "idle_slow", "rate_slow") or c.get("check"):
            res["outputs_match"] = self.check_outputs()
        return res

    def cmd_accuracy(self, c):
        d = np.load(c["npz"])
        n = int(c.get("n", 1000))
        x, y = d["x_test"][:n], d["y_test"][:n]
        b = self.batch
        correct = 0
        for i in range(0, n - n % b, b):
            self.b.set_input(x[i:i + b])
            self.b.run(b)
            correct += int((self.b.output().reshape(-1) == y[i:i + b]).sum())
        done = n - n % b
        self.b.set_input(make_input(self.b))   # restore the fixed golden input
        return {"n": done, "top1": round(100.0 * correct / max(done, 1), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    agent = Agent(MockBoard() if args.mock else Board())
    print(json.dumps({"ready": True, "mock": args.mock, "t": time.time()}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            if c["cmd"] == "quit":
                print(json.dumps({"bye": True}), flush=True)
                break
            out = getattr(agent, "cmd_" + c["cmd"])(c)
            out.update({"ok": out.get("ok", True), "cmd": c["cmd"]})
        except Exception as e:   # report and keep serving
            out = {"ok": False, "cmd": line[:80], "error": "%s: %s" % (type(e).__name__, e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
