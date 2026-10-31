import sys, os
from pypdf import PdfReader
for p in sys.argv[1:]:
    r = PdfReader(p)
    out = os.path.splitext(p)[0] + ".txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, page in enumerate(r.pages):
            f.write(f"\n=== page {i+1} ===\n")
            f.write(page.extract_text() or "")
    print(os.path.basename(p), len(r.pages), "pages", os.path.getsize(out), "chars")
