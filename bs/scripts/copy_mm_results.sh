#!/usr/bin/env bash
# Copy training run summaries (result.json, test_pred.csv) from a run root into the project results folder.
# Usage: copy_mm_results.sh RUN_ROOT DEST_SUBDIR
set -uo pipefail
RUN="${1:-$HOME/bs/mm_runs_full}"
SUB="${2:-mm_runs_full}"
DST="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/$SUB"
mkdir -p "$DST"
for d in "$RUN"/*/; do
  n=$(basename "$d")
  [ -f "$d/result.json" ] && cp "$d/result.json" "$DST/${n}_result.json"
  [ -f "$d/test_pred.csv" ] && cp "$d/test_pred.csv" "$DST/${n}_test_pred.csv"
done
for f in "$RUN"/*.json "$RUN"/*.csv; do [ -f "$f" ] && cp "$f" "$DST/"; done
ls "$DST" | wc -l
