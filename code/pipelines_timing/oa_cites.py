"""Forward citations in OpenAlex: python -I oa_cites.py <out.json> "<title of the cited work>" """
import json, sys, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
def get(url):
    return json.load(urllib.request.urlopen(url + "&mailto=research@example.org", timeout=60))
hit = get("https://api.openalex.org/works?per-page=3&search=" + urllib.parse.quote(sys.argv[2]))["results"]
if not hit:
    print("not found"); sys.exit()
w = hit[0]
print("cited work:", w["id"], w.get("publication_year"), w["title"][:100], "| cited_by", w.get("cited_by_count"))
oid = w["id"].split("/")[-1]
out = []
for page in (1, 2, 3):
    r = get(f"https://api.openalex.org/works?per-page=50&page={page}&filter=cites:{oid}&sort=publication_date:desc")
    for x in r["results"]:
        ab = x.get("abstract_inverted_index") or {}
        words = sorted(((p, k) for k, ps in ab.items() for p in ps))
        out.append({"title": x.get("title"), "date": x.get("publication_date"), "doi": x.get("doi"),
                    "venue": ((x.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "abstract": " ".join(k for _, k in words)[:1500], "oa": (x.get("open_access") or {}).get("oa_url")})
    if len(r["results"]) < 50:
        break
json.dump(out, open(sys.argv[1], "w"), indent=1)
for o in out:
    print(o["date"], "|", (o["venue"] or "")[:25], "|", (o["title"] or "")[:110])
