# Déploiement de l'endpoint promptgen (RunPod Serverless, SANS volume)

Objectif : un endpoint **autonome** qui exécute `handler.py` — décode une image, fait **une
passe Qwen3-VL**, renvoie le prompt Qwen (JSON) **dans la réponse du job** → écrit sur ton
laptop par `promptgen/promptgen_runpod.py` (`make prompt`). Miroir « côté texte » de
`comfyui/serverless/`.

> **Prérequis / parité** : Qwen3-VL exige un `transformers` récent. Le `Dockerfile` épingle
> `transformers>=4.57.0` comme **point de départ à valider**. Le smoke test de fin de build
> (chargement du modèle) échoue s'il faut une autre version — ajuste alors le pin et rebuild.

## Coûts (même pool GPU que la génération d'image)

Qwen3-VL-8B tient sur un GPU **≥ 24 Go** (pool Ampere 24, comme l'endpoint image). Réglages
coût-optimaux, identiques : **min workers 0**, **idle ~5 s**, **FlashBoot on**, **pas de
network volume** (modèle baké → 0 $ de stockage), sortie dans la réponse (texte léger, pas de
S3). Une passe ≈ quelques secondes à chaud ; le poste de coût = les cold starts → enchaîne.

## Build de l'image (modèle baké)

Le build télécharge ~16 Go de poids → il se fait **côté RunPod ou sur une machine Docker**,
pas depuis ce chat.

**Voie recommandée — Import Git Repository (RunPod build pour toi)**
1. Pousser le contenu de `promptgen/serverless/` dans un repo GitHub (comme
   `comfyui/serverless/` l'a été).
2. Console RunPod → **Serverless → New Endpoint → Import Git Repository** → ce repo.
   - Contexte de build = la racine du repo (le `Dockerfile` y référence `serverless/…` ;
     si tu pousses le contenu du dossier à plat, ajuste les chemins `COPY` en conséquence).
   - Build args (optionnels) : `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct`, `CUDA_TAG=…`.
   - RunPod build l'image (téléchargement du modèle au build — plusieurs minutes).
3. GPU **≥ 24 Go**, container disk ≥ 30 Go, **min workers 0**, FlashBoot on.
4. Récupérer l'**Endpoint ID** → `fakinflu/.env` (`RUNPOD_PROMPTGEN_ENDPOINT_ID`).

**Voie alternative — build local + push** (si tu as Docker + un registre)
```bash
cd promptgen                 # contexte = promptgen/ (le Dockerfile COPY serverless/…)
docker build -f serverless/Dockerfile -t <toncompte>/fakinflu-promptgen:v1 .
docker push <toncompte>/fakinflu-promptgen:v1
# puis créer l'endpoint serverless sur cette image (console RunPod ou API)
```

## Vérifier / lancer

```bash
cd fakinflu
make prompt-check                          # endpoint joignable, aucun credit
make prompt IMG=references/source.png      # image -> data/work/last_prompt.txt
```

1er appel = cold start (chargement du modèle baké, quelques minutes) ; ensuite quelques
secondes. Réponse worker : `{"output": {"prompt": "...", "variants": [...], "fiche": {...}}}`.

## Alternative de serving

Ce kit utilise un **handler custom** (`runpod` SDK + `transformers`) — le plus fidèle au
« une requête, une passe » et au contrat I/O sur mesure (image + keep/subject/avoid → prompt).
Pour du débit élevé plus tard, on pourra repartir du **worker vLLM** RunPod (endpoint
OpenAI-compatible), au prix d'un contrat I/O générique à ré-encapsuler.

## Pièges

- **smoke test échoue au build** : `transformers` trop ancien pour Qwen3-VL → monte le pin.
- **OOM au chargement** : GPU < 24 Go → prends un GPU plus gros (comme l'endpoint image).
- **404 au `make prompt-check`** : mauvais `RUNPOD_PROMPTGEN_ENDPOINT_ID`.
- **sortie non-JSON** : le handler renvoie alors `{"prompt": <texte brut>, "_unparsed": true}` —
  resserre `system_prompt.txt` si ça arrive souvent.
