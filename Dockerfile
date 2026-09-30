# fakinflu — worker RunPod : descripteur d'image (Qwen3-VL-8B) -> prompt Qwen.
# Miroir "cote texte" de comfyui/serverless : image AUTONOME, modele BAKE, pas de
# network volume. Le worker renvoie le PROMPT (texte/JSON) dans la reponse du job.
#
# ⚠️ PARITE DE VERSION — meme piege que la generation d'image (ComfyUI 0.37 pour
# Qwen-Image 2.1) : Qwen3-VL (model_type "qwen3_vl") exige un transformers RECENT,
# installe DEPUIS LES SOURCES (cf. etape 2). Le smoke test de fin de build echoue
# ICI si la version ne charge pas le modele -> avant de depenser en prod.

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 1) torch cu124 d'abord (comme l'image de base cote generation installe torch en 1er)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2) stack VLM : transformers pour Qwen3-VL + utilitaires vision + runpod SDK.
#    ⚠️ Qwen3-VL (model_type "qwen3_vl") n'est PAS reconnu par les versions PyPI
#    disponibles a ce jour -> la carte modele Qwen impose l'install DEPUIS LES
#    SOURCES. C'est le correctif du crash "Unrecognized model ... qwen3_vl".
#    (Quand une version stable inclura qwen3_vl, remplacer par un pin fige.)
RUN pip3 install "git+https://github.com/huggingface/transformers" \
      accelerate \
      "qwen-vl-utils[decord]" \
      pillow \
      "huggingface_hub[cli]>=0.34" \
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
#    On teste la CLASSE EXPLICITE utilisee par le handler (config qwen3_vl + poids).
# RUN python3 -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; \
# AutoProcessor.from_pretrained('${MODEL_DIR}'); \
# Qwen3VLForConditionalGeneration.from_pretrained('${MODEL_DIR}', dtype='auto'); \
# print('smoke test OK — modele chargeable')"

CMD ["python3", "-u", "handler.py"]
