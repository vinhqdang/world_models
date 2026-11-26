import subprocess,sys,urllib.request,urllib.parse,re,time
def q(query,n=6,idlist=None):
    if idlist: url="https://export.arxiv.org/api/query?id_list="+idlist+"&max_results=50"
    else: url="https://export.arxiv.org/api/query?search_query="+urllib.parse.quote(query)+f"&max_results={n}&sortBy=relevance"
    for t in range(3):
        try:
            x=subprocess.run(["curl","-s","-m","40",url],capture_output=True,text=True).stdout
            if "<feed" in x: break
            time.sleep(4)
        except Exception as e: x=""; time.sleep(5)
    out=[]
    for e in x.split("<entry>")[1:]:
        i=re.search(r"<id>https?://arxiv.org/abs/(.*?)</id>",e).group(1)
        t=re.sub(r"\s+"," ",re.search(r"<title>(.*?)</title>",e,re.S).group(1))
        au=re.findall(r"<name>(.*?)</name>",e)
        p=re.search(r"<published>(.*?)</published>",e).group(1)[:10]
        out.append(f"  {i} | {p} | {t} | {', '.join(au[:3])}")
    print("Q:",idlist or query); print("\n".join(out) if out else "  (none)"); sys.stdout.flush(); time.sleep(3.1)
if __name__=="__main__":
    for a in sys.argv[1:]:
        if a.startswith("id:"): q(None,idlist=a[3:])
        else: q(a)
