#!/usr/bin/env bash
# Telecharge le modele VLM (Qwen3-VL) dans un dossier local, pour le BAKER dans l'image.
# Usage (build) :
#   MODEL_ID=Qwen/Qwen3-VL-8B-Instruct MODEL_DIR=/models/qwen3vl-8b ./download_model.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/models/qwen3vl-8b}"

echo ">> telechargement de ${MODEL_ID} -> ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# huggingface-hub est tire par transformers ; on utilise le CLI hf pour un download robuste.
pip3 install --no-cache-dir "huggingface_hub[cli]>=0.34" >/dev/null

hf download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --exclude "*.pth" "*.bin" "original/*"    # on garde les safetensors

echo ">> OK. Contenu :"
find "${MODEL_DIR}" -maxdepth 1 -type f -printf '   %p  (%s octets)\n' | head -n 40
