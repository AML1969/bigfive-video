#!/usr/bin/env bash
# Start (or restart) the BS Profiler 3.0 UI preview server on :7882 with a finished job (default: the newest one).
# Usage: run_preview.sh [JOB_DIR] [PORT]      Stop: pkill -f "ui_previe[w]"
pkill -f "ui_previe[w].py" 2>/dev/null; sleep 2
PORT="${2:-7882}"
mkdir -p "$HOME/bs3_data/logs"; cd "$HOME"
export GRADIO_TEMP_DIR="$HOME/bs3_data/gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"
export PYTHONWARNINGS=ignore GRADIO_ANALYTICS_ENABLED=False no_proxy="localhost,127.0.0.1,0.0.0.0"
setsid nohup "$HOME/bs/venv/bin/python" /mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs3-studio/scripts/ui_preview.py "$@" \
  > "$HOME/bs3_data/logs/preview.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$PORT/" || true)
  [ "$code" = "200" ] && break
done
grep -aE "preview job|refused" "$HOME/bs3_data/logs/preview.log"
echo "http://localhost:$PORT -> HTTP ${code:-000}"
