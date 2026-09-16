#!/usr/bin/env bash
# serve ~/ui_check on :7872 (static UI check pages)
pkill -f "http[.]server 7872" 2>/dev/null; sleep 1
setsid nohup python3 -m http.server 7872 --directory "$HOME/ui_check" > /tmp/ui_check.log 2>&1 < /dev/null &
sleep 2
curl -s -o /dev/null -w "static server %{http_code}
" http://localhost:7872/dark.html
