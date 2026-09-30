#!/usr/bin/env python3
import base64, io, json, os

import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# --- Cached models HF (Runpod) ------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline: n'ira jamais télécharger au runtime (échouera si pas caché)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/system_prompt.txt")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

def resolve_snapshot_path(model_id: str) -> str:
    """
    Helper Runpod (docs) : résout le chemin snapshot local du modèle cached.
    """
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")

LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)

with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# --- chargement unique au cold start ------------------------------------------
_model = Qwen3VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
)
_model.eval()

_proc = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
)

def _b64_to_image(b64: str) -> Image.Image:
    if "," in b64[:64] and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _user_text(inp: dict) -> str:
    lang = inp.get("lang") or "anglais"
    parts = [f"Langue du prompt final : {lang}."]
    for key, label in (
        ("keep", "A garder absolument"),
        ("subject", "Sujet souhaite"),
        ("scene", "Scene souhaitee"),
        ("avoid", "A eviter"),
    ):
        v = inp.get(key)
        if v:
            parts.append(f"{label} : {v}")
    parts.append("Variantes demandees : " + ("oui (3)" if inp.get("variants") else "non"))
    return "\n".join(parts)

def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(raw[i : j + 1])
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
    except Exception as e:
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