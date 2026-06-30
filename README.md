# Artcurve AI — Art Authenticity Detector

AI-generated art detection service for [ArtCurve](https://github.com/CypherAli/Artcurve_Be) — distinguishes hand-made artwork from AI-generated images (Stable Diffusion / Latent Diffusion).

> Training and the inference API run on a Coder GPU workspace. This local checkout mirrors the code pushed to [github.com/CypherAli/Artcurve_AI](https://github.com/CypherAli/Artcurve_AI).

## How it works

Fine-tunes CLIP ViT-B/32 with LoRA adapters on the [AI-ArtBench dataset](https://www.kaggle.com/datasets/ravidussilva/real-ai-art) (155K images: 50K real classical paintings + 105K AI-generated across 10 art styles).

```
Image → CLIP ViT-B/32 (frozen) → LoRA adapter (trainable) → Linear head → [AI_GENERATED | ORIGINAL]
```

Only 360K of 87.8M parameters are trainable (0.4%) — fast fine-tuning on a single T4 GPU (~30 min for 3 epochs).

## Results

Evaluated on the AI-ArtBench **test split** (30,000 images, never seen during training or validation):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| AI_GENERATED | 0.9999 | 1.0000 | 1.0000 | 20,000 |
| ORIGINAL | 1.0000 | 0.9998 | 0.9999 | 10,000 |
| **Overall accuracy** | | | **99.99%** | 30,000 |

Confusion matrix: 2 false negatives out of 30,000, 0 false positives.

**Caveat:** the AI samples are exclusively from Stable Diffusion / Latent Diffusion (2022-era models). The classifier has not been tested against newer generators (Midjourney v6, DALL-E 3, Flux, SDXL) — expect lower real-world accuracy until the dataset is expanded.

## Files

| File | Purpose |
|------|---------|
| `prepare_data.py` | Symlinks the raw AI-ArtBench folders into a binary `real/` vs `ai/` structure |
| `train.py` | Fine-tunes CLIP + LoRA, saves best checkpoint by validation accuracy |
| `test_model.py` | Quick sanity check on random held-out samples |
| `api_server.py` | FastAPI inference server (`POST /detect`) |

## Setup

```bash
pip install torch torchvision transformers peft datasets Pillow scikit-learn kagglehub fastapi uvicorn python-multipart

python3 -c "import kagglehub; print(kagglehub.dataset_download('ravidussilva/real-ai-art'))"

python3 prepare_data.py

python3 train.py
```

## Running the API

```bash
python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### `GET /health`
```json
{ "status": "ok", "device": "cuda" }
```

### `POST /detect`
Multipart form upload, field name `file`.

```bash
curl -X POST http://localhost:8000/detect -F "file=@artwork.jpg"
```

Response:
```json
{
  "label": "AI_GENERATED",
  "confidence": 0.94,
  "predictions": { "AI_GENERATED": 0.94, "ORIGINAL": 0.06 }
}
```

## Model weights

Trained weights (`best_model.pt`, 352MB) are not committed to this repo (too large for git). Hosted on Hugging Face Hub: **[Hoang2509/artcurve-ai-detector](https://huggingface.co/Hoang2509/artcurve-ai-detector)**.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="Hoang2509/artcurve-ai-detector", filename="best_model.pt")
```

## Integration with ArtCurve

The main ArtCurve backend has a prepared `artwork_type` field (`ORIGINAL | AI_GENERATED | AI_ASSISTED`) that artists self-declare on upload. This service is called during the `AI_MODERATING` state to cross-check the declaration — mismatches are flagged for manual review rather than auto-rejected, since detector accuracy on modern AI art is unverified.

## Roadmap

- [ ] Expand dataset with Midjourney v6 / Flux / SDXL samples for broader generalization
- [ ] Export to ONNX for CPU-only inference (avoid keeping a GPU running 24/7)
- [ ] Add perceptual-hash + CLIP-embedding duplicate/plagiarism detection (separate from AI-vs-real classification)
