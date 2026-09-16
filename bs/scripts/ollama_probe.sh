#!/usr/bin/env bash
# Find how to reach the Windows Ollama server from WSL: localhost (mirrored networking) or the host IP (NAT).
set -uo pipefail
HOSTIP=$(ip route show default | awk '{print $3}')
echo "default gateway (Windows host in NAT mode): $HOSTIP"
for url in "http://localhost:11434" "http://$HOSTIP:11434" "http://$(hostname).local:11434"; do
  code=$(curl -s --max-time 4 -o /dev/null -w "%{http_code}" "$url/api/tags" || true)
  echo "$url -> HTTP $code"
done
echo "WSL networking mode: $(grep -i networkingMode /mnt/c/Users/nrsmh/.wslconfig 2>/dev/null || echo 'not set (NAT)')"
