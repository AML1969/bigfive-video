"""Probe which Ollama request settings make a vision model return the JSON ratings.
Usage: rater_probe.py VIDEO [MODEL]
"""
import json
import sys
import time
import urllib.request

from bs_bigfive.backend_mm import MMBackend, MMConfig, default_ollama_url
from llm_rater import PROMPT

video = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "qwen3-vl:30b"
url = default_ollama_url()
be = MMBackend(MMConfig(behavior_frames=16))
images = be._sample_frames_jpeg(video)
prompt = PROMPT.format(n=16, transcript_note="") + "\n\nTranscript: \"(none)\""


def call(endpoint, payload):
    req = urllib.request.Request(f"{url}/api/{endpoint}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d, time.time() - t


opts = {"num_predict": 400, "temperature": 0.0}
tests = [
    ("generate json+think=false", "generate", {"model": model, "prompt": prompt, "images": images, "stream": False,
                                               "format": "json", "think": False, "options": opts}),
    ("chat json+think=false", "chat", {"model": model, "stream": False, "format": "json", "think": False, "options": opts,
                                       "messages": [{"role": "user", "content": prompt, "images": images}]}),
    ("chat think=false", "chat", {"model": model, "stream": False, "think": False, "options": opts,
                                  "messages": [{"role": "user", "content": prompt, "images": images}]}),
]
for name, ep, payload in tests:
    try:
        d, secs = call(ep, payload)
        msg = d.get("message", {}) if ep == "chat" else d
        resp = msg.get("response", "") if ep == "generate" else msg.get("content", "")
        think = msg.get("thinking", "")
        print(f"== {name}: {secs:.1f}s done={d.get('done_reason')} | response={resp[:250]!r} | thinking={str(think)[:250]!r}")
    except Exception as e:
        print(f"== {name}: ERROR {e}")
