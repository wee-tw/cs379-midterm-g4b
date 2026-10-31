"""Measurement agent for PYNQ-Z2 (runs ON THE BOARD as root: inside the venv on PYNQ v3.0.1, or
with the system Python 3.6 / numpy 1.13 on PYNQ v2.5; compat_v25/ tests the latter).

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
                 (the rate windows take k: k frames per start, one start every k/rate s;
                 k > 1 is batched race-to-halt, which spreads the pipeline fill over k)
     jit         back-to-back batches at the current clock, which the PC sets just high
                 enough for the rate (streaming at a just-in-time clock); the CPU sleeps
                 through most of each batch instead of polling the DMA
  probe_throttle                        find a stop/restart sequence that works twice in a
                                        row; checks outputs after
  verify_throttle counts?               replay the stop with LAST_CNT = n at the slowest
                                        divider: the status bit must clear after about n / f
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
# The same throttle on Zynq UltraScale+ (UG1087 PLx_THR_CTRL/CNT) documents RUNNING as the
# *counter* status, LAST_CNT as "the number of clocks after receiving a trigger that the
# clocks should stop", CPU_START as needing LAST_CNT programmed first, and CNT_RST = 1 as
# "held in reset" (its reset value, with the clock free running). So a cleared bit can also
# mean "counter held in reset, clock running"; verify_throttle checks the counting itself.
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

    def start(self, batch):
        self.accel.execute_on_buffers(asynch=True, batch_size=batch)

    def wait(self):
        self.accel.wait_until_finished()                   # polls the output DMA's ap_done

    def set_input(self, x):
        a = self.accel
        a.copy_input_data_to_device(a.pack_input(a.fold_input(x)))

    def clear_output(self):
        """Zero the device output buffer, so a check cannot pass on stale results."""
        buf = self.accel.obuf_packed_device[0]
        buf[:] = 0
        buf.flush()

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
        # throttle model: CPU_START is edge-triggered; LAST_CNT > 0 stops the clock after
        # that many FCLK0 pulses; CNT_RST = 1 holds the counter (status bit clear) while the
        # clock keeps running, which is the misleading case verify_throttle must catch
        self.regs = {FPGA0_THR_CTRL: 0, FPGA0_THR_CNT: 0}
        self.stop_at = None
        self.mhz, self.accel, self.batch = 100.0, None, 1

    def clock_running(self):
        return self.stop_at is None or time.perf_counter() < self.stop_at

    def rd(self, off):
        if off == FPGA0_THR_STA:
            return 0 if self.regs[FPGA0_THR_CTRL] & 2 or not self.clock_running() else STA_RUNNING
        return self.regs.get(off, 0)

    def wr(self, off, val):
        prev = self.regs.get(off, 0)
        self.regs[off] = val
        if off == FPGA0_THR_CTRL:
            if val & 2:
                self.stop_at = None
            elif val & 1 and not prev & 1:
                cnt = self.regs[FPGA0_THR_CNT]
                self.stop_at = time.perf_counter() + cnt / (self.mhz * 1e6) if cnt else None

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
        assert self.clock_running(), "PL access while FCLK0 is stopped"
        time.sleep(batch * self.cycles / (self.mhz * 1e6) + 1e-4)

    def start(self, batch):
        assert self.clock_running(), "PL access while FCLK0 is stopped"
        self.t_done = time.perf_counter() + batch * self.cycles / (self.mhz * 1e6) + 1e-4

    def wait(self):
        time.sleep(max(0.0, self.t_done - time.perf_counter()))

    def set_input(self, x):
        self.x = x

    def clear_output(self):
        pass

    def output(self):
        n = self.x.shape[0] if self.x is not None else self.batch
        return np.asarray([[int(v) % 10] for v in
                           (self.x.reshape(n, -1).sum(1) if self.x is not None else range(n))])

    def ishape(self):
        return (self.batch, 32, 32, 3)

    def idt(self):
        return None


def make_input(board, seed=0):
    """Fixed random input in the accelerator's input range (UINT8 images, INT8 for KWS).
    RandomState, not default_rng: the PYNQ v2.5 image has numpy 1.13 (default_rng is 1.17+)."""
    rng = np.random.RandomState(seed)
    dt = board.idt()
    lo, hi = (0, 255) if dt is None else (int(dt.min()), int(dt.max()))
    t = np.int8 if lo < 0 and hi <= 127 else np.uint8 if hi <= 255 else np.int32
    return rng.randint(lo, hi + 1, size=board.ishape()).astype(t)


def platform_info():
    """Software stack of the board, recorded with every session (PYNQ v2.5 and v3.0.1 differ)."""
    import platform
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "kernel": platform.release(), "board": os.environ.get("BOARD")}
    try:
        import pynq
        info["pynq"] = getattr(pynq, "__version__", None)
    except Exception as e:   # mock on a PC, or pynq without root
        info["pynq"] = "unavailable (%s)" % type(e).__name__
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                info["os"] = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        info["os"] = None
    return info


class Agent:
    def __init__(self, board):
        self.b = board
        self.stop_seq = self.start_seq = None
        self.halt_verified = None
        self.halts = self.halt_fail = 0
        self.stuck = False
        self.golden = None
        self.batch = 1

    # ---- clock throttle -------------------------------------------------------------
    def running(self):
        return bool(self.b.rd(FPGA0_THR_STA) & STA_RUNNING)

    def apply(self, seq):
        for off, val in seq:
            self.b.wr(off, val)

    def halt(self, timeout=0.02):
        """Stop FCLK0; counts a failure if the status bit does not clear within timeout."""
        self.apply(STOP_SEQS[self.stop_seq])
        self.halts += 1
        t_end = time.perf_counter() + timeout
        while self.running():
            if time.perf_counter() > t_end:
                self.halt_fail += 1
                return False
        return True

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

    def check_outputs(self, k=None):
        """Run the fixed input (first k images; all by default) and compare with golden."""
        k = min(int(k or self.batch), self.batch)
        self.b.clear_output()
        self.b.run(k)
        out = self.b.output()
        return bool(np.array_equal(out[:k], self.golden[:k]))

    # ---- commands -------------------------------------------------------------------
    def cmd_info(self, c):
        return {"t": time.time(), "fclk0_mhz": self.b.get_mhz(), "temp_c": die_temp_c(),
                "regs": self.regs(), "platform": platform_info(),
                "stop_seq": self.stop_seq, "start_seq": self.start_seq,
                "halt_verified": self.halt_verified, "stuck": self.stuck}

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
        self.stop_seq = self.start_seq = None
        self.halt_verified = None
        for sname, sseq in STOP_SEQS.items():
            ok_cycles = 0
            for cycle in range(2):   # a usable pair must also stop again after its restart
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
                tried.append({"stop": sname, "cycle": cycle, "stopped": stopped,
                              "sta_after_stop": sta, "restart": restarted})
                if restarted is None:
                    self.stuck = True
                    return {"ok": False, "tried": tried, "before": before, "stuck": True,
                            "note": "clock did not restart: no PL access; reload or power-cycle"}
                if not stopped:
                    break
                ok_cycles += 1
                rname_ok = restarted
            if ok_cycles == 2:
                self.stop_seq, self.start_seq = sname, rname_ok
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

    def cmd_verify_throttle(self, c):
        """Check, with PS registers only, that the stop sequence really gates FCLK0.

        At the slowest divider (~0.25 MHz) the stop sequence is replayed with LAST_CNT = n
        instead of 1. If the throttle counts FCLK0 pulses, the status bit stays set for about
        n / f and then clears, so n / t_stop must match f for both n. A bit that clears at
        once (counter held in reset, clock still running) fails. The PL is touched only
        after the restart (output check).
        """
        if not self.stop_seq or self.stuck:
            raise RuntimeError("no working throttle sequence; run probe_throttle first")
        f_run = self.b.get_mhz()
        f_low = self.b.set_slowest()
        time.sleep(0.05)
        trials = []
        for n in c.get("counts", [20000, 40000]):
            seq = [(off, n if off == FPGA0_THR_CNT and val == 1 else val)
                   for off, val in STOP_SEQS[self.stop_seq]]
            expected = n / (f_low * 1e6)
            t0 = time.perf_counter()
            self.apply(seq)
            t_stop, polls, sta_first = None, 0, self.b.rd(FPGA0_THR_STA)
            while time.perf_counter() - t0 < 3 * expected + 0.5:
                polls += 1
                if not self.running():
                    t_stop = time.perf_counter() - t0
                    break
            sta_end = self.b.rd(FPGA0_THR_STA)
            self.resume()
            trials.append({"last_cnt": n, "expected_s": round(expected, 5),
                           "t_stop_s": None if t_stop is None else round(t_stop, 5),
                           "count_mhz": None if not t_stop else round(n / t_stop / 1e6, 4),
                           "sta_first": hex(sta_first), "sta_end": hex(sta_end),
                           "polls": polls})
        self.b.set_mhz(f_run)
        self.halt_verified = all(t["count_mhz"] and 0.5 < t["count_mhz"] / f_low < 2.0
                                 for t in trials)
        out_ok = self.check_outputs()
        return {"ok": out_ok, "halt_verified": self.halt_verified, "slowest_mhz": f_low,
                "restored_mhz": self.b.get_mhz(), "trials": trials, "outputs_match": out_ok}

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
        self.halts = self.halt_fail = 0
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
        elif state == "jit":             # streaming at a just-in-time clock: frames back to
            b = min(int(c.get("batch", self.batch)), self.batch)   # back, CPU asleep while
            t_img = None                 # a batch runs; a 1/10 batch first times one image,
            while time.time() < end:     # so the busy wait at the start stays short (never
                bb = b if t_img else max(1, b // 10)   # read the DMA status ourselves:
                t1 = time.time()                       # ap_done is cleared on read)
                self.b.start(bb)
                if t_img:
                    time.sleep(0.95 * t_img * bb)
                self.b.wait()
                t_img = (time.time() - t1) / bb
                n += bb
                if rate:                 # keep the average at R when even the slowest
                    ahead = t0 + n / rate - time.time()   # FCLK0 is faster than needed
                    if ahead > 0:
                        time.sleep(min(ahead, max(0.0, end - time.time())))
        elif state == "single":          # back-to-back batch-1 runs (MLPerf single-stream)
            while time.time() < end:
                self.b.run(1)
                n += 1
        elif state in ("rate", "rate_halt", "rate_slow"):
            k = int(c.get("k", 1))       # frames per start (batched race-to-halt when > 1)
            period, nxt = k / rate, time.time()
            # cpu "sleep" (default): after timing the first frame, sleep through 90% of each
            # frame before polling the DMA, so the CPU is mostly idle as in the jit windows;
            # "poll" keeps the FINN driver's busy wait (measures that overhead)
            cpu, t_frame = c.get("cpu", "sleep"), None
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
                if cpu == "sleep" and t_frame:
                    self.b.start(k)
                    time.sleep(0.9 * t_frame)
                    self.b.wait()
                else:
                    t1 = time.time()
                    self.b.run(k)
                    t_frame = t_frame or time.time() - t1
                if state == "rate_halt":
                    self.halt()
                elif state == "rate_slow":
                    self.b.set_slowest()
                n += k
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
        if state.startswith("rate"):
            res["cpu"] = c.get("cpu", "sleep")
            res["k"] = int(c.get("k", 1))
        if "halt" in state:
            res.update({"halts": self.halts, "halt_fail": self.halt_fail,
                        "halt_verified": self.halt_verified})
        if state in ("idle_halt", "rate_halt", "idle_slow", "rate_slow", "jit") or c.get("check"):
            # at a just-in-time clock a full batch can take seconds: check the window's batch
            res["outputs_match"] = self.check_outputs(
                c.get("batch") if state == "jit" else c.get("k") if state.startswith("rate")
                else None)
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
