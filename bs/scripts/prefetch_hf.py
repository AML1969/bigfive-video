"""Prefetch Hugging Face model repos into the local cache (weights + configs, no .bin duplicates when safetensors exist).

Usage: prefetch_hf.py REPO_ID [REPO_ID ...]
"""
import sys
import time

from huggingface_hub import snapshot_download

PATTERNS = {
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim": ["*.json", "*.bin", "*.safetensors"],
    # main branch of the CLIP repo has no safetensors, only pytorch_model.bin
    "openai/clip-vit-base-patch32": ["*.json", "*.txt", "*.bin"],
}
DEFAULT = ["*.json", "*.txt", "*.safetensors", "*.model"]

for rid in sys.argv[1:]:
    t = time.time()
    for attempt in range(1, 4):
        try:
            p = snapshot_download(rid, allow_patterns=PATTERNS.get(rid, DEFAULT), max_workers=2)
            print(f"{rid}: {p} ({time.time() - t:.0f}s)", flush=True)
            break
        except Exception as e:  # network hiccup: retry, resume .incomplete files
            print(f"{rid}: attempt {attempt} failed: {e}", flush=True)
            time.sleep(10)
print("prefetch done", flush=True)
