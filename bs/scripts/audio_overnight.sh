#!/usr/bin/env bash
# Start the audio-encoder comparison at night: wait until START_HHMM (default 01:00) and until the GPU has been idle
# (no other bs processes computing) for 5 minutes, so it does not fight with the web UI. Usage: audio_overnight.sh [HH:MM]
set -uo pipefail
START="${1:-01:00}"
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
cd "$HOME"
target=$(date -d "today $START" +%s)
[ "$target" -le "$(date +%s)" ] && target=$(date -d "tomorrow $START" +%s)
echo "waiting until $(date -d @"$target" '+%Y-%m-%d %H:%M') ..."
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done
idle=0
while [ "$idle" -lt 5 ]; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${util:-100}" -lt 10 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
echo "GPU idle, starting at $(date +%H:%M:%S)"
bash "$S/audio_candidates.sh" 2>&1 | cut -c1-200
