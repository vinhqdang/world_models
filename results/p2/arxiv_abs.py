import sys, urllib.request, re, time
ids = sys.argv[1:]
url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids) + f"&max_results={len(ids)}"
t = urllib.request.urlopen(url, timeout=40).read().decode()
for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
    i = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
    ti = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1))
    ab = re.sub(r"\s+", " ", re.search(r"<summary>(.*?)</summary>", e, re.S).group(1))
    d = re.search(r"<published>(.*?)</published>", e).group(1)[:10]
    print(f"[{i}] {d} {ti}\n  {ab[:1100]}\n")
