#!/usr/bin/env bash
# Stage 1 environment for BS (Big Five from video) inside WSL2 Ubuntu 24.04.
# Creates ~/bs/venv with torch cu130 (RTX 5090, sm_120) and OCEAN-AI + pinned deps.
set -euo pipefail
VENV="$HOME/bs/venv"
mkdir -p "$HOME/bs/models" "$HOME/bs/eval" "$HOME/bs/logs"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
"$PIP" install -U pip wheel setuptools
# torch 2.11 triple: the newest torch that still has a matching torchaudio (oceanai imports torchaudio at module level).
"$PIP" install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# OCEAN-AI without its dependency pins (they drag jupyterlab==4.2.5 and torch>=2.2.2 resolution problems)
"$PIP" install oceanai==1.0.0a48 --no-deps
# Proven-compatible vision/audio set (same pins as MM-PSYCHE requirements) + OCEAN-AI runtime deps
"$PIP" install \
  numpy==1.26.4 scipy pandas requests \
  opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86 mediapipe==0.10.21 \
  opensmile librosa audioread soundfile scikit-learn \
  liwc sentencepiece sacremoses "transformers==4.45.1" \
  ipython jupyterlab "gradio==5.8.0" pillow toml tqdm
echo "===== verify ====="
"$VENV/bin/python" - <<'PY'
import torch, torchvision, torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) != 0)
import numpy, cv2, mediapipe, opensmile, transformers, librosa
print("numpy", numpy.__version__, "cv2", cv2.__version__, "mediapipe", mediapipe.__version__, "transformers", transformers.__version__)
from oceanai.modules.lab.build import Run
print("oceanai Run import ok")
PY
echo "===== setup done ====="
