import json, sys, urllib.request, urllib.parse, time
sys.stdout.reconfigure(encoding="utf-8")
kind = sys.argv[1]
for q in sys.argv[2:]:
    url = f"https://api.github.com/search/{kind}?q=" + urllib.parse.quote(q) + "&per_page=8"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "curl"})
    try:
        d = json.load(urllib.request.urlopen(req))
    except Exception as e:
        print("==", q, "ERR", e); continue
    print("==", q, d.get("total_count"))
    for it in d.get("items", []):
        if kind == "repositories":
            print(f"  {it['stargazers_count']:5d} {it['pushed_at'][:10]} {it['full_name']}: {(it.get('description') or '')[:110]}")
        else:
            print("  ", it["repository"]["full_name"], it["path"])
    time.sleep(7)
