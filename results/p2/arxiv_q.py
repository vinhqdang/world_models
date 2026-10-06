import sys, urllib.parse, urllib.request, re, time
def q(query, n=8):
    url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query) + f"&start=0&max_results={n}&sortBy=submittedDate&sortOrder=descending"
    for _ in range(3):
        try:
            t = urllib.request.urlopen(url, timeout=30).read().decode()
            break
        except Exception as e:
            t = ""; time.sleep(3)
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        i = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
        ti = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1))
        out.append(f"  {i} | {ti}")
    return out
if __name__ == "__main__":
    for qq in sys.argv[1:]:
        print("Q:", qq)
        print("\n".join(q(qq)))
        time.sleep(3)
