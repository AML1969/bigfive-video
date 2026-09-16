#!/usr/bin/env bash
# Start the web UI inside WSL (detached, survives the launching shell). Open http://localhost:7860 on Windows.
# Usage: run_web.sh [PORT]      Stop: pkill -f "bs web"
set -uo pipefail
PORT="${1:-7860}"
mkdir -p "$HOME/bs/logs"
cd "$HOME"
export PYTHONWARNINGS=ignore
export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"
export GRADIO_ANALYTICS_ENABLED=False
# a just-killed server can linger for a few seconds; wait for it instead of reporting "already running"
for i in $(seq 1 10); do
  pgrep -f "bs.web" >/dev/null || break
  if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/"; then echo "already running and answering on :$PORT"; exit 0; fi
  sleep 2
done
pgrep -f "bs.web" >/dev/null && { echo "old server still shutting down; try again"; exit 1; }
setsid nohup "$HOME/bs/venv/bin/bs" web --port "$PORT" > "$HOME/bs/logs/web.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/" || true)
  [ "$code" = "200" ] && break
done
tail -3 "$HOME/bs/logs/web.log" | grep -vE "SyntaxWarning|invalid escape"
echo "http://localhost:$PORT -> HTTP ${code:-000} after $((i*2))s"
