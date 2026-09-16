"""LLM-as-rater experiment: a local vision-language model (Ollama) rates Big Five directly from sampled frames
and the transcript. Writes one JSON per clip (resumable) and a prediction csv in the standard layout
(Path + Openness..Non-Neuroticism [+ Interview]).

Usage: llm_rater.py EVAL_DIR OUT_CSV [--model qwen3-vl:30b] [--frames 16] [--limit N] [--no-transcript]
EVAL_DIR holds <stem>.mp4 and <stem>.txt (transcript).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from bs_bigfive.backend_mm import MMBackend, MMConfig, default_ollama_url
from bs_bigfive.norms import OCEANAI_COLUMNS

KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability", "interview"]

PROMPT = """You are rating APPARENT personality: the first impression an audience of ordinary observers forms about the person in this video, exactly as crowd annotators did for the ChaLearn First Impressions dataset. You see {n} frames sampled uniformly from a ~15-second clip of a person talking to the camera{transcript_note}.

For each Big Five trait, give the share of observers (0.00-1.00) who would perceive the person as HIGH on that trait: 0.50 means a typical person, values below 0.30 or above 0.70 are rare and need clear evidence. Traits:
- openness: curious, imaginative, expressive, unconventional
- conscientiousness: organized, reliable, careful, composed
- extraversion: energetic, talkative, outgoing, smiling, animated
- agreeableness: warm, friendly, cooperative, pleasant
- emotional_stability: calm, relaxed, confident (the opposite of neurotic/anxious)
Also give "interview": the share of observers who would invite this person to a job interview after this clip.

Base your ratings on facial expressions, smiling, eye contact, energy, gestures, posture, grooming, voice-related cues visible in speech content and fluency of the transcript. Do not default to 0.50 for everything; discriminate between traits.

Return ONLY a JSON object with exactly these six keys: "openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability", "interview". Each value is a number between 0 and 1 with two decimals that reflects YOUR rating of this particular person (never a placeholder)."""


def rate(url, model, images, transcript, n_frames, use_transcript=True):
    note = " and the transcript of what they say" if use_transcript else ""
    prompt = PROMPT.format(n=n_frames, transcript_note=note)
    if use_transcript:
        prompt += f"\n\nTranscript: \"{(transcript or '').strip()[:1500]}\""
    payload = {"model": model, "prompt": prompt, "images": images, "stream": False, "think": False, "format": "json",
               "keep_alive": "30m", "options": {"num_predict": 200, "temperature": 0.0, "num_ctx": 32768}}
    req = urllib.request.Request(f"{url}/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode("utf-8"))
    # qwen3-vl through Ollama puts the JSON into the "thinking" field even with think=false
    text = data.get("response", "") or str(data.get("thinking", "") or "")
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
    out = {}
    for k in KEYS:
        v = obj.get(k, obj.get(k.replace("_", "-"), None))
        try:
            out[k] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            out[k] = None
    return out, text, data.get("eval_duration", 0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--model", default="qwen3-vl:30b")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=640, help="longest side of the frames sent to the model (px)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-transcript", action="store_true")
    a = ap.parse_args()
    d = Path(a.eval_dir)
    variant = "" if (a.frames, a.max_side) == (16, 640) else f"_f{a.frames}_s{a.max_side}"   # keep the old cache valid
    cache = d / f"rater_{a.model.replace(':', '_').replace('/', '_')}{'_novtx' if a.no_transcript else ''}{variant}"
    cache.mkdir(exist_ok=True)
    be = MMBackend(MMConfig(behavior_frames=a.frames, behavior_max_side=a.max_side))   # only for frame sampling
    url = default_ollama_url()
    clips = sorted(d.glob("*.mp4"))
    if a.limit:
        clips = clips[: a.limit]
    rows, t0, done, bad = [], time.time(), 0, 0
    for i, p in enumerate(clips, 1):
        cj = cache / (p.stem + ".json")
        if cj.exists():
            res = json.loads(cj.read_text(encoding="utf-8"))
        else:
            txt = p.with_suffix(".txt")
            transcript = txt.read_text(encoding="utf-8") if txt.exists() else ""
            scores, raw, secs = rate(url, a.model, be._sample_frames_jpeg(p), transcript, a.frames, not a.no_transcript)
            res = {"scores": scores, "raw": raw, "seconds": secs}
            cj.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            done += 1
        s = res["scores"]
        if any(s.get(k) is None for k in KEYS[:5]):
            bad += 1
            continue
        rows.append({"Path": p.name, **{c: s[k] for k, c in zip(KEYS[:5], OCEANAI_COLUMNS)}, "Interview": s.get("interview")})
        if i % 25 == 0:
            print(f"{i}/{len(clips)} ({done} new, {bad} unparsable, {(time.time() - t0) / max(1, done):.1f}s/clip)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"wrote {a.out_csv}: {len(df)} clips, {bad} unparsable, model={a.model}, frames={a.frames}, url={url}")
    print(df[OCEANAI_COLUMNS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
