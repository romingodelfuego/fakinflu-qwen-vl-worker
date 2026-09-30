#!/usr/bin/env bash
# Telecharge le modele VLM (Qwen3-VL) dans un dossier local, pour le BAKER dans l'image.
# Usage (build) :
#   MODEL_ID=Qwen/Qwen3-VL-8B-Instruct MODEL_DIR=/models/qwen3vl-8b ./download_model.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/models/qwen3vl-8b}"

echo ">> telechargement de ${MODEL_ID} -> ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# 'hf' est fourni par huggingface_hub (installe a l'etape 2 du Dockerfile).
# NE PAS utiliser 'huggingface-cli' : deprecated et cassé dans huggingface_hub >= 1.x.
hf download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --exclude "*.pth" "original/*"

# Verification : sans ca, un download partiel passe inapercu (build "COMPLETED")
# et le worker crashe au runtime (OSError: no file named model.safetensors).
count=$(find "${MODEL_DIR}" -name '*.safetensors' | wc -l | tr -d ' ')
if [ "$count" -eq 0 ] || [ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]; then
  echo "!! ERREUR : safetensors incomplets dans ${MODEL_DIR} (count=$count)"
  find "${MODEL_DIR}" -maxdepth 1 -type f
  exit 1
fi

echo ">> OK. ${count} shard(s) safetensors."
ls -lh "${MODEL_DIR}"/*.safetensors
