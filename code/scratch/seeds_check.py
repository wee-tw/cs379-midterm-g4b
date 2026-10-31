import glob, json, os, sys
for d in ("exp3", "exp6", "exp7"):
    for f in sorted(glob.glob(os.path.join(sys.argv[1], d, "sens_base*.json"))):
        a = json.load(open(f))
        print(d, os.path.basename(f), "seed", a["args"]["seed"], "acc", a["final_acc"])
