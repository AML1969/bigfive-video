#!/usr/bin/env bash
# Download the publicly available SSL-MEPR unimodal checkpoints (text: Google Drive, audio+scene: Yandex Disk).
# Face and body checkpoints were never published (Drive folder is empty), so the app can run on 3 of 5 modalities.
set -uo pipefail
DST="${1:-$HOME/bs/ssl_mepr_ckpt}"
mkdir -p "$DST/text" "$DST/audio" "$DST/scene"
gd() { # id, out
  curl -sL --retry 3 "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t" -o "$2"
}
yd() { # public_key, filename, out
  href=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=$1&path=%2F$2" | python3 -c "import sys,json;print(json.load(sys.stdin)[\"href\"])")
  curl -sL --retry 3 "$href" -o "$3"
}
gd 1RglGB6mmm4g-zJBwQbsrjgTTol4zO1Hf "$DST/text/Mamba_bge-small_emotion.pt"
gd 1z3Fc3MYW_NhYQSq2A6-GHbdmuE3fG-it "$DST/text/Mamba_Transformer_bge-small_fusion.pt"
gd 1yNOiipjZ5linJR3lXfkNv1gqcmmclSLu "$DST/text/Transformer_bge-small_personality.pt"
AUD="https%3A%2F%2Fdisk.yandex.ru%2Fd%2FAYFC4OWdUAnS1g"
yd "$AUD" best_fusion_overall_mamba.pt   "$DST/audio/best_fusion_overall_mamba.pt"
yd "$AUD" final_best_model_uni_mamba.pt  "$DST/audio/final_best_model_uni_mamba.pt"
yd "$AUD" best_mamba_regressor.pth       "$DST/audio/best_mamba_regressor.pth"
SCN="https%3A%2F%2Fdisk.360.yandex.ru%2Fd%2FN5T8uvutLRy0Xg"
yd "$SCN" clip_fusion_transformer_transformer_mamba_best_model_dev.pt "$DST/scene/clip_fusion_transformer_transformer_mamba_best_model_dev.pt"
echo "===== downloaded ====="
ls -la "$DST"/*/
