"""python -I vgrep.py <name (file stem in lit_txt)> <regex> [<regex> ...]: context of each match."""
import re, sys
sys.stdout.reconfigure(encoding="utf-8")
base = r"C:/Users/wee/AppData/Local/Temp/claude/D--115-1-embedded-sys-midterm-report/0ab647fd-9b9f-46c9-8993-886b1139c5f2/scratchpad/lit_txt/"
t = re.sub(r"\s+", " ", open(base + sys.argv[1] + ".txt", encoding="utf-8", errors="replace").read())
for pat in sys.argv[2:]:
    ms = list(re.finditer(pat, t, re.I))
    print(f"## /{pat}/ x{len(ms)}")
    for m in ms[:3]:
        print("   ..." + t[max(0, m.start() - 170): m.end() + 170] + "...")
