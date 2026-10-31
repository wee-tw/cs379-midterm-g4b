"""OpenAlex keyword sweep: python -I oa_search.py <out.json> "<query>" [from_date]"""
import json, sys, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
q, since = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "2024-01-01")
url = ("https://api.openalex.org/works?per-page=40&search=" + urllib.parse.quote(q)
       + "&filter=from_publication_date:" + since + "&mailto=research@example.org")
data = json.load(urllib.request.urlopen(url, timeout=60))
out = []
for w in data.get("results", []):
    ab = w.get("abstract_inverted_index") or {}
    words = sorted(((p, k) for k, ps in ab.items() for p in ps))
    out.append({"id": w["id"], "doi": w.get("doi"), "title": w.get("title"), "date": w.get("publication_date"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "cited_by": w.get("cited_by_count"), "abstract": " ".join(k for _, k in words)[:1500],
                "oa": (w.get("open_access") or {}).get("oa_url")})
json.dump(out, open(sys.argv[1], "w"), indent=1)
for o in out:
    print(o["date"], (o["venue"] or "")[:30], "|", (o["title"] or "")[:110])
