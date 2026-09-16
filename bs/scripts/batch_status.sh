#!/usr/bin/env bash
# Status of the Russian batch: per-video state, running processes, GPU. Usage: batch_status.sh [OUT_DIR]
OUT="${1:-$HOME/bs/ru_runs}"
for d in "$OUT"/*/; do
  n=$(basename "$d")
  if [ -f "$d/result.json" ]; then
    echo "$n: done ($(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r.get('segments'),'segs',r['timings_sec'].get('total_wall'),'s')" "$d/result.json"))"
  elif [ -f "$d/failed.txt" ]; then
    echo "$n: FAILED: $(tr '\n' ';' < "$d/failed.txt")"
  else
    echo "$n: in progress ($(ls "$d/segments" 2>/dev/null | grep -c '\.mp4$') segments cut, $(ls "$d/segments" 2>/dev/null | grep -c '\.json$') json)"
  fi
done
echo "--- processes"
pgrep -af 'batch_ru|bs web|debug_segment' | grep -v pgrep | cut -c1-140
echo "--- gpu"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- time $(date +%H:%M:%S)"
