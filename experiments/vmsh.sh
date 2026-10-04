#!/bin/bash
# run a shell command on the Colab VM session: vmsh.sh '<cmd>'
export PATH="$HOME/.local/bin:$PATH"
python3 - "$1" > /tmp/_vmcmd.py <<'PY'
import sys, json
print("import subprocess,sys\np=subprocess.Popen(%s,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)\nfor l in p.stdout: print(l,end='',flush=True)\np.wait()" % json.dumps(sys.argv[1]))
PY
colab exec -s "${VMS:-wm}" -f /tmp/_vmcmd.py 2>&1
