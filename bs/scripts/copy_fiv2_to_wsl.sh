#!/usr/bin/env bash
# Copy the FIV2 clips from the Windows download folder to ext4 (~30 GB) and lay them out the MM-PSYCHE way:
#   ~/data/fiv2/video/{train,dev,test}/<name>.mp4   (HF "validation" -> "dev")
# Only complete files (*.mp4, no .part) are copied; rsync makes it resumable.
set -uo pipefail
SRC="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/data/first-impressions-v2"
DST="$HOME/data/fiv2/video"
mkdir -p "$DST/train" "$DST/dev" "$DST/test"
start=$(date +%s)
rsync -a --include='*.mp4' --exclude='*' "$SRC/train/" "$DST/train/"
rsync -a --include='*.mp4' --exclude='*' "$SRC/validation/" "$DST/dev/"
rsync -a --include='*.mp4' --exclude='*' "$SRC/test/" "$DST/test/"
for s in train dev test; do echo "$s: $(ls "$DST/$s" | wc -l) files"; done
du -sh "$DST"
echo "copy done in $(( $(date +%s) - start ))s"
