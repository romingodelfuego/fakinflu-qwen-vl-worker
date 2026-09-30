ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 1) Torch (identique à ton fichier actuel)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2) Stack VLM
# NOTE: tu peux garder l'install transformers depuis les sources si tu en as besoin
# pour "model_type=qwen3_vl".
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install \
      "git+https://github.com/huggingface/transformers" \
      accelerate \
      "qwen-vl-utils[decord]" \
      pillow \
      "huggingface_hub>=0.34" \
      runpod

# 3) Plus de BAKE du modèle (Runpod le cache via le champ "Model" de l'endpoint)
# 4) Fichiers app
COPY system_prompt.txt /app/system_prompt.txt
COPY handler.py        /app/handler.py

CMD ["python3", "-u", "handler.py"]