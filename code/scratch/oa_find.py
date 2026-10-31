import json, sys, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(q) + "&per-page=3"
    for w in json.load(urllib.request.urlopen(url))["results"]:
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        oa = (w.get("open_access") or {}).get("oa_url")
        print(w["publication_year"], "|", w["display_name"][:110], "|", w.get("doi"), "|", src, "| OA:", oa)
        print("   ", ", ".join(a["author"]["display_name"] for a in w["authorships"][:5]))
    print()
