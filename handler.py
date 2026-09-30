#!/usr/bin/env python3
# =====================================================================
# fakinflu - handler RunPod serverless : IMAGE -> PROMPT Qwen (descripteur VLM)
#
# Miroir "cote texte" du worker de generation d'image :
#   - image AUTONOME, modele Qwen3-VL BAKE (pas de network volume)
#   - une passe par requete, resultat (texte) renvoye dans la reponse du job
#
# Contrat (voir ../README.md) :
#   input : {"image": <b64>, "keep"?, "subject"?, "scene"?, "avoid"?,
#            "lang"? (defaut "anglais"), "variants"? (bool)}
#   output: {"prompt": "...", "variants": [...], "fiche": {...}}
#
# Le "cerveau" (methode selective : extraction 10-dim + assemblage) vit dans
# system_prompt.txt, bake a cote de ce handler. On l'edite sans retoucher le code.
# =====================================================================
import base64, io, json, os

import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3vl-8b")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/system_prompt.txt")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# --- chargement unique au cold start (reste chaud entre les jobs) ------------
# Classe explicite Qwen3-VL (l'auto-mapping ne la resout de facon fiable qu'avec
# un transformers tres recent ; cf. Dockerfile : install depuis les sources).
_model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype="auto", device_map="auto")
_model.eval()
_proc = AutoProcessor.from_pretrained(MODEL_DIR)


def _b64_to_image(b64: str) -> Image.Image:
    if "," in b64[:64] and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]           # tolere un data: URL
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _user_text(inp: dict) -> str:
    lang = inp.get("lang") or "anglais"
    parts = [f"Langue du prompt final : {lang}."]
    for key, label in (("keep", "A garder absolument"),
                       ("subject", "Sujet souhaite"),
                       ("scene", "Scene souhaitee"),
                       ("avoid", "A eviter")):
        v = inp.get(key)
        if v:
            parts.append(f"{label} : {v}")
    parts.append("Variantes demandees : " + ("oui (3)" if inp.get("variants") else "non"))
    return "\n".join(parts)


def _parse(raw: str) -> dict:
    """Le system prompt impose du JSON ; on parse, avec repli si le modele bavarde."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except json.JSONDecodeError:
                pass
    return {"prompt": raw, "variants": [], "fiche": {}, "_unparsed": True}


def handler(job):
    inp = job.get("input", {}) or {}
    b64 = inp.get("image")
    if not b64:
        return {"error": "champ 'image' (base64) requis"}

    try:
        image = _b64_to_image(b64)
    except Exception as e:                       # noqa: BLE001
        return {"error": f"image base64 invalide : {e}"}

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _user_text(inp)},
        ]},
    ]

    text = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = _proc(text=[text], images=[image], return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = out[:, batch["input_ids"].shape[1]:]
    raw = _proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    data = _parse(raw)
    data.setdefault("variants", [])
    data.setdefault("fiche", {})
    return data


runpod.serverless.start({"handler": handler})
