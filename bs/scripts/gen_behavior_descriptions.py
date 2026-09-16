"""Generate behaviour descriptions for clips with a local Ollama vision model (paper prompt, sampled frames).

Writes <stem>.behavior.txt next to each clip (resumable; existing files are skipped) and, when the MM-PSYCHE
csv is given, prints a side-by-side sample against the paper's Qwen3-VL-4B descriptions.

Usage: gen_behavior_descriptions.py VIDEO_DIR [--n N] [--model qwen3-vl:30b] [--frames 16] [--csv test_full_with_description.csv]
"""
import argparse
import csv
import time
from pathlib import Path

from bs_bigfive.backend_mm import MMBackend, MMConfig

ap = argparse.ArgumentParser()
ap.add_argument("video_dir")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--model", default="qwen3-vl:30b")
ap.add_argument("--frames", type=int, default=16)
ap.add_argument("--csv", default=None)
ap.add_argument("--show", type=int, default=3)
a = ap.parse_args()

be = MMBackend(MMConfig(ollama_model=a.model, behavior_frames=a.frames))   # describe_behavior needs no checkpoint
clips = sorted(Path(a.video_dir).glob("*.mp4"))
if a.n:
    clips = clips[: a.n]
ref = {}
if a.csv:
    ref = {r["video_name"]: r["text_llm"] for r in csv.DictReader(open(a.csv, encoding="utf-8"))}
t0, done, shown = time.time(), 0, 0
for i, p in enumerate(clips, 1):
    out = p.with_suffix(".behavior.txt")
    if out.exists() and out.stat().st_size > 0:
        continue
    t = time.time()
    text = be.describe_behavior(p)
    out.write_text(text, encoding="utf-8")
    done += 1
    if shown < a.show:
        shown += 1
        print(f"\n== {p.name} ({time.time() - t:.1f}s, {len(text.split())} words)\nOLLAMA: {text}")
        if p.stem in ref:
            print(f"PAPER : {ref[p.stem]}")
    if i % 25 == 0:
        print(f"{i}/{len(clips)} clips, {done} generated, {(time.time() - t0) / max(1, done):.1f}s per clip", flush=True)
print(f"done: {done} generated, {len(clips) - done} skipped, {time.time() - t0:.0f}s total, url={be.cfg.ollama_url or 'auto'}")
