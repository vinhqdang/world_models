#!/bin/bash
# usage: ax.sh "query" [n]
q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$1")
curl -s --max-time 40 "https://export.arxiv.org/api/query?search_query=$q&max_results=${2:-8}&sortBy=relevance" | python3 -c "
import sys,re
x=sys.stdin.read()
for e in x.split('<entry>')[1:]:
    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]
    t=re.sub(r'\s+',' ',re.search(r'<title>(.*?)</title>',e,re.S).group(1))
    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]
    a=re.findall(r'<name>(.*?)</name>',e)
    print(i,p,'|',t,'|',', '.join(a[:3]))
"
