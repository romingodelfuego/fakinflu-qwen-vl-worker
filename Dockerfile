# fakinflu — worker RunPod : descripteur d'image (Qwen3-VL-8B) -> prompt Qwen.
# Miroir "cote texte" de comfyui/serverless : image AUTONOME, modele BAKE, pas de
# network volume. Le worker renvoie le PROMPT (texte/JSON) dans la reponse du job.
#
# ⚠️ PARITE DE VERSION — meme piege que la generation d'image (ComfyUI 0.37 pour
# Qwen-Image 2.1) : Qwen3-VL exige une version RECENTE de transformers. Le tag
# epingle ci-dessous est un point de depart A VALIDER : le smoke test de fin de
# build echoue ICI si la version ne charge pas le modele.

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 1) torch cu124 d'abord (comme l'image de base cote generation installe torch en 1er)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2) stack VLM : transformers RECENT pour Qwen3-VL + utilitaires vision + runpod SDK
#    (>=4.57.0 = point de depart ; ajuster au 1er build selon le support Qwen3-VL)
RUN pip3 install \
      "transformers>=4.57.0" \
      accelerate \
      "qwen-vl-utils[decord]" \
      pillow \
      "huggingface_hub[cli]>=0.34" \
      hf_transfer \
      runpod

# 3) BAKE du modele
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
ENV MODEL_DIR=/models/qwen3vl-8b
COPY download_model.sh /tmp/download_model.sh
RUN MODEL_ID=${MODEL_ID} MODEL_DIR=${MODEL_DIR} bash /tmp/download_model.sh

# 4) cerveau (methode selective) + handler
COPY system_prompt.txt /app/system_prompt.txt
COPY handler.py        /app/handler.py

# 5) smoke test : le modele se charge-t-il avec cette version de transformers ?
#    (echoue au BUILD, pas en prod — comme --quick-test-for-ci cote ComfyUI)
#RUN python3 -c "from transformers import AutoProcessor, AutoModelForImageTextToText; \
#AutoProcessor.from_pretrained('${MODEL_DIR}'); \
#AutoModelForImageTextToText.from_pretrained('${MODEL_DIR}', torch_dtype='auto'); \
#print('smoke test OK — modele chargeable')"

CMD ["python3", "-u", "handler.py"]
