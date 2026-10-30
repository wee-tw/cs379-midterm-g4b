"""Run a queue of train_cifar.py jobs, a few at a time on one GPU, resumably.

  python run_queue.py jobs.json --parallel 2

jobs.json is a list of {"name": ..., "args": [...train_cifar.py arguments...],
"after": [names that must finish first]}. Each job writes <out>/<name>.json (its log),
<name>.pt and <name>.out; a job whose .json already exists is skipped, so the queue can
be stopped and restarted. A summary table is printed when the queue is empty.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--out", required=True)
    p.add_argument("--parallel", type=int, default=2)
    args = p.parse_args()
    jobs = json.load(open(args.jobs))
    os.makedirs(args.out, exist_ok=True)
    done = lambda j: os.path.exists(os.path.join(args.out, j["name"] + ".json"))
    running, failed = {}, set()
    while True:
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "finished", name, "exit", proc.returncode,
                      flush=True)
                del running[name]
                if not done({"name": name}):
                    failed.add(name)
        pending = [j for j in jobs if not done(j) and j["name"] not in running
                   and j["name"] not in failed]
        ready = [j for j in pending
                 if all(done({"name": a}) for a in j.get("after", []))]
        # jobs waiting on a failed job can never start
        ready_or_waiting = [j for j in pending
                            if not any(a in failed for a in j.get("after", []))]
        while ready and len(running) < args.parallel:
            j = ready.pop(0)
            base = os.path.join(args.out, j["name"])
            cmd = [sys.executable, os.path.join(HERE, "train_cifar.py")] + [
                a.replace("{out}", args.out) for a in j["args"]] + [
                "--save", base + ".pt", "--log", base + ".json"]
            log = open(base + ".out", "w")
            running[j["name"]] = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            print(time.strftime("%H:%M:%S"), "started", j["name"], flush=True)
        if not running and not ready and not ready_or_waiting:
            break
        if not running and not ready:
            break  # everything left depends on something that failed
        time.sleep(10)
    print("\nname, final_acc, best_acc, wall_sec")
    for j in jobs:
        f = os.path.join(args.out, j["name"] + ".json")
        if os.path.exists(f):
            r = json.load(open(f))
            print(f"{j['name']}, {r['final_acc']}, {r['best_acc']}, {r['wall_sec']}")
        else:
            print(f"{j['name']}, FAILED or skipped")


if __name__ == "__main__":
    main()
