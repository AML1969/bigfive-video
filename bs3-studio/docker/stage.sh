#!/bin/bash
# Build contexts for the two images of BS Profiler 3.1, made on the build machine (WSL) from git and the local caches.
#   bash docker/stage.sh [OUT]          default OUT=/mnt/c/bs_docker  (outside OneDrive: the contexts are ~12 GB)
# Result: OUT/ctx_app (Dockerfile, req/, models/, app/, VERSION) and OUT/ctx_ollama (Dockerfile, models/).
# Nothing personal goes in: the code comes from `git archive HEAD` (tracked files only), no jobs, no videos.
set -euo pipefail
OUT=${1:-/mnt/c/bs_docker}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
VENV=$HOME/bs/venv
HF=$HOME/.cache/huggingface/hub
OLLAMA_STORE=${OLLAMA_STORE:-/mnt/c/Users/$USER/.ollama/models}
# the Hugging Face models the product and OCEAN-AI load (bs3/mm/extractors.py, bs3/analyses, bs3/translate.py,
# bs3/settings.py ASR_MODEL, oceanai/modules/lab/text.py)
MODELS="openai/whisper-large-v3-turbo openai/clip-vit-base-patch32 laion/clap-htsat-fused
michellejieli/emotion_text_classifier audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim trpakov/vit-face-expression
Helsinki-NLP/opus-mt-ru-en Helsinki-NLP/opus-mt-en-ru google-bert/bert-base-multilingual-cased"

A=$OUT/ctx_app
O=$OUT/ctx_ollama
rm -rf "$A" "$O"
mkdir -p "$A/app" "$A/req" "$A/models/hf/hub" "$A/models/mm_runs_seeds" "$O/models/manifests" "$O/models/blobs"

echo "== code: git archive HEAD bs3-studio"
COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
# the image gets exactly the committed code; only the docker/ folder itself may be ahead of the commit
test -z "$(git -C "$REPO" status --short -- bs3-studio ':!bs3-studio/docker')" || { echo "bs3-studio has uncommitted changes: commit first"; exit 1; }
git -C "$REPO" archive HEAD bs3-studio | tar -x -C "$A/app"
printf 'BS Profiler 3.1\ncode commit %s (branch %s)\nbuilt %s\n' "$COMMIT" "$(git -C "$REPO" branch --show-current)" "$(date -u +%Y-%m-%dT%H:%MZ)" > "$A/VERSION"
cp "$A/VERSION" "$A/app/bs3-studio/VERSION"

echo "== Python packages: requirements + constraints from the tested venv"
cp "$REPO/bs3-studio/requirements-wsl.txt" "$A/req/"
"$VENV/bin/pip" freeze --exclude-editable 2>/dev/null \
  | grep -vE '^(-e |#)| @ |^(bs-bigfive|bs2-studio|bs3-studio)==' > "$A/req/constraints.txt"
echo "   constraints: $(wc -l < "$A/req/constraints.txt") pinned packages"

echo "== Hugging Face models (the snapshot of refs/main, symlinks resolved)"
for id in $MODELS; do
  d="$HF/models--${id//\//--}"
  rev=$(cat "$d/refs/main")
  mkdir -p "$A/models/hf/hub/models--${id//\//--}/refs" "$A/models/hf/hub/models--${id//\//--}/snapshots"
  # no newline: huggingface_hub reads the commit hash verbatim, "hash\n" names a snapshot that does not exist
  printf %s "$rev" > "$A/models/hf/hub/models--${id//\//--}/refs/main"
  cp -rL "$d/snapshots/$rev" "$A/models/hf/hub/models--${id//\//--}/snapshots/$rev"
  echo "   $id  $(du -sh "$A/models/hf/hub/models--${id//\//--}" | cut -f1)"
done

echo "== OCEAN-AI weights"
cp -r "$HOME/bs/models" "$A/models/oceanai"
echo "   $(du -sh "$A/models/oceanai" | cut -f1)"

echo "== AMLAI 1.0 weights"
for s in seed1 seed2 seed3 seed4 seed42; do
  mkdir -p "$A/models/mm_runs_seeds/$s"
  cp "$HOME/bs/mm_runs_seeds/$s/best.pt" "$A/models/mm_runs_seeds/$s/"
done
(cd "$A/models/mm_runs_seeds" && sha256sum seed*/best.pt > SHA256SUMS && cat SHA256SUMS)

echo "== Ollama model qwen2.5vl:7b from $OLLAMA_STORE"
MAN=manifests/registry.ollama.ai/library/qwen2.5vl/7b
mkdir -p "$O/models/$(dirname "$MAN")"
cp "$OLLAMA_STORE/$MAN" "$O/models/$MAN"
python3 - "$O/models/$MAN" <<'PY' | while read -r dg; do cp "$OLLAMA_STORE/blobs/$dg" "$O/models/blobs/$dg"; done
import json, sys
m = json.load(open(sys.argv[1]))
for layer in m["layers"] + [m["config"]]:
    print(layer["digest"].replace(":", "-"))
PY
echo "   $(du -sh "$O/models" | cut -f1)"

cp "$HERE/Dockerfile" "$A/Dockerfile"
cp "$HERE/ollama/Dockerfile" "$O/Dockerfile"
echo "== done: code $COMMIT"
du -sh "$A" "$O"
