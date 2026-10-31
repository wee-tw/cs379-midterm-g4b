import json, sys, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(q) + "&per-page=1"
    w = json.load(urllib.request.urlopen(url))["results"][0]
    inv = w.get("abstract_inverted_index") or {}
    words = sorted((p, t) for t, ps in inv.items() for p in ps)
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    print(w["publication_year"], "|", w["display_name"], "|", w.get("doi"), "|", src)
    print("  authors:", ", ".join(a["author"]["display_name"] for a in w["authorships"]))
    print("  abstract:", " ".join(t for _, t in words) or "(none)")
    print()
