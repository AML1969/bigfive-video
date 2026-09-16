#!/usr/bin/env bash
# Pre-download OCEAN-AI weight files (Google Drive) into the models cache so the library skips them.
# Usage: prefetch_oceanai_weights.sh [MODELS_DIR] [URL_LIST] [KEY_REGEX]
# The URL list (scripts/oceanai_weight_urls.txt) has lines "<corpus-key> <url>", extracted from oceanai core.py.
set -uo pipefail
DST="${1:-$HOME/bs/models}"
LIST="${2:-$(dirname "$0")/oceanai_weight_urls.txt}"
KEYS="${3:-^(audio|video|text|avt)/(fi|mupta)/}"
mkdir -p "$DST"
cd "$DST"
while read -r key url; do
  [[ "$key" =~ $KEYS ]] || continue
  [[ "$key" == */fe ]] && continue      # 98 MB ResNet50: leave to the library / separate run
  name=$(curl -sIL --max-time 30 "$url" | grep -ioE 'filename="?[^";]+' | tail -1 | sed -E 's/filename="?//')
  if [ -z "$name" ]; then echo "$key -> no filename (skipped)"; continue; fi
  if [ -f "$name" ]; then echo "$key -> $name exists"; continue; fi
  curl -sL --retry 3 --max-time 600 "$url" -o "$name.part" && mv "$name.part" "$name" && echo "$key -> $name $(stat -c %s "$name") bytes"
done < "$LIST"
echo "===== $DST ====="
ls -la "$DST" | grep -vE "^total|^d"
