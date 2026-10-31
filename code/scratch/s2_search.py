"""Semantic Scholar relevance search: python -I s2_search.py <out.json> "<query>" [year range e.g. 2020-2026]"""
import json, sys, time, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
q = sys.argv[2]; yr = sys.argv[3] if len(sys.argv) > 3 else "2019-2026"
url = ("https://api.semanticscholar.org/graph/v1/paper/search?limit=40&year=" + yr + "&query=" + urllib.parse.quote(q)
       + "&fields=title,year,venue,abstract,openAccessPdf,externalIds,citationCount")
for attempt in range(4):
    try:
        data = json.load(urllib.request.urlopen(url, timeout=60)); break
    except Exception as e:
        print("retry", e); time.sleep(8)
else:
    sys.exit(1)
out = data.get("data", [])
json.dump(out, open(sys.argv[1], "w"), indent=1)
for o in out:
    print(o.get("year"), "|", (o.get("venue") or "")[:28], "|", o.get("citationCount"), "|", (o.get("title") or "")[:110])
