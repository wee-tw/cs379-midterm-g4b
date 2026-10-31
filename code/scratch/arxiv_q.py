import sys, urllib.request, urllib.parse, re, html
sys.stdout.reconfigure(encoding="utf-8")
for q in sys.argv[1:]:
    url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + "&max_results=8&sortBy=submittedDate&sortOrder=descending"
    t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    print("==", q)
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        idv = re.search(r"<id>(.*?)</id>", e).group(1).rsplit("/", 1)[1]
        pub = re.search(r"<published>(.*?)</published>", e).group(1)[:10]
        print(f"  {pub} {idv} {html.unescape(title)[:120]}")
