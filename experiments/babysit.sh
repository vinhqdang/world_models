#!/bin/bash
# Keep the Colab session alive and pull every finished checkpoint / result file to the local scratch dir.
# usage: babysit.sh <remote_dir> <local_dir> <done_marker_relative>
export PATH="$HOME/.local/bin:$PATH"
R=$1; L=$2; D=$3
mkdir -p $L
while true; do
  files=$(colab exec -s wm -f <(echo "import glob;print('\n'.join(sorted(glob.glob('$R/*.pt')+glob.glob('$R/*.json'))))") 2>/dev/null | grep -E "\.(pt|json)$")
  for f in $files; do
    b=$(basename $f); [ -f $L/$b ] || { colab download -s wm $f $L/$b >/dev/null 2>&1 && echo "pulled $b"; }
  done
  if colab exec -s wm -f <(echo "import os;print('FIN' if os.path.exists('$R/$D') else 'RUN')") 2>/dev/null | grep -q FIN; then echo FINISHED; break; fi
  sleep 120
done
