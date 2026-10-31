"""Cross-check: every number of the final main table must appear where the docs report it."""
import json, re, sys
R = "D:/115-1/embedded sys midterm report"
t = json.load(open(f"{R}/results/search/main_table_tsr_v1218_m0.json"))
docs = {n: open(f"{R}/docs/{n}", encoding="utf-8").read() for n in ("results_digest.md", "research_overview.md", "implementation_notes.md")}
sec0 = docs["results_digest.md"].split("### 10/31 03:30 版")[0]
main = docs["research_overview.md"].split("### 主結果（連續推論")[1].split("### 被選中設計")[0]
notes = docs["implementation_notes.md"].split("### 12.44")[1]
missing = []
def need(text, where, val):
    if val not in text:
        missing.append((where, val))
for m in t:
    for p in ("nominal", "z2_anchor"):
        r = t[m][p]
        e = f"{r['A_verified']['E_mJ']:.3f}"
        b1 = f"{100 * r['B1_measured']['rel_to_A']:+.1f}%".lstrip("+")
        b2 = f"{100 * r['B2_measured']['rel_to_A']:+.1f}%".lstrip("+")
        b3 = f"{100 * r['B3_measured']['rel_to_A']:+.1f}%".lstrip("+")
        for where, text in (("digest §0", sec0), ("notes 12.44", notes)):
            need(text, where, e); need(text, where, b1); need(text, where, b2)
        need(notes, "notes 12.44 (B3)", b3)
        if p == "nominal":
            need(main, "overview main table", e)
        need(main, "overview main table", b1); need(main, "overview main table", b2)
print("missing:", missing if missing else "none")
