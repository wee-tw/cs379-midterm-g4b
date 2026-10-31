import sys, glob, os
sys.path.insert(0, sys.argv[1])
sys.stdout.reconfigure(encoding="utf-8")
from pynq.pl_server import hwh_parser as hp
for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.hwh"))):
    h = hp.HWH(f)
    ips = {k: (hex(v["phys_addr"]), v.get("type")) for k, v in h.ip_dict.items()}
    print(os.path.basename(f), "| IPs:", ips)
    print("   clocks:", {k: v for k, v in h.clock_dict.items() if v.get("enable")})
