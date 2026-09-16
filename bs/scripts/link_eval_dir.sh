#!/usr/bin/env bash
# Make a sibling eval folder that links the clips, Rev transcripts and labels of SRC but has NO behaviour
# descriptions, so a different description generator can write its own <stem>.behavior.txt there.
# Usage: link_eval_dir.sh SRC DST
set -uo pipefail
SRC="$1"; DST="$2"
mkdir -p "$DST"
n=0
for mp4 in "$SRC"/*.mp4; do
  stem=$(basename "$mp4" .mp4)
  [ -e "$DST/$stem.mp4" ] || ln -s "$(readlink -f "$mp4")" "$DST/$stem.mp4"
  [ -e "$DST/$stem.txt" ] || { [ -f "$SRC/$stem.txt" ] && ln -s "$(readlink -f "$SRC/$stem.txt")" "$DST/$stem.txt"; }
  n=$((n+1))
done
[ -e "$DST/labels.csv" ] || ln -s "$(readlink -f "$SRC/labels.csv")" "$DST/labels.csv"
echo "$DST: $n clips linked, $(ls "$DST"/*.behavior.txt 2>/dev/null | wc -l) descriptions present"
