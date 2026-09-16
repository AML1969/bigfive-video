#!/usr/bin/env bash
# Show raw LLM-rater responses cached for an eval folder. Usage: inspect_rater_cache.sh CACHE_DIR [N]
set -uo pipefail
DIR="${1:-$HOME/bs/eval/fi_test200/rater_qwen3-vl_30b}"
N="${2:-3}"
pkill -f "bs.web" 2>/dev/null; sleep 1
echo "web procs left: $(pgrep -fc 'bs.web' || true)"
for f in $(ls "$DIR"/*.json | head -n "$N"); do
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(sys.argv[1].split("/")[-1], "| secs", round(d.get("seconds", 0), 1), "| scores", d["scores"])
print("RAW:", repr(d["raw"][:500]))
PY
done
