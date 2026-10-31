import json, sys, time, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
for q in sys.argv[1:]:
    url = "https://api.github.com/search/issues?per_page=15&q=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "curl"})
    try:
        d = json.load(urllib.request.urlopen(req))
    except Exception as e:
        print("==", q, "ERR", e)
        time.sleep(8)
        continue
    print("==", q, "total", d.get("total_count"))
    for it in d.get("items", []):
        kind = "PR" if "pull_request" in it else "issue"
        repo = it["repository_url"].split("repos/")[1]
        print(f"  {it['created_at'][:10]} {kind:5s} {repo}#{it['number']} [{it['state']}] {it['title'][:100]}")
    time.sleep(8)
