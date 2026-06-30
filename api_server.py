import torch, torch.nn as nn, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from PIL import Image

app = FastAPI(title="ArtCurve AI Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["AI_GENERATED", "ORIGINAL"]  # alphabetical: ai=0, real=1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),
])

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
vision = get_peft_model(clip.vision_model, LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],lora_dropout=0.1))

class D(nn.Module):
    def __init__(s,v):
        super().__init__()
        s.v=v; s.h=nn.Sequential(nn.Linear(768,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,2))
    def forward(s,x): return s.h(s.v(pixel_values=x).pooler_output)

model = D(vision).to(device)
model.load_state_dict(torch.load("/home/coder/artcurve-ai/model/best_model.pt", map_location=device))
model.eval()

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred_idx = probs.argmax().item()
    return {
        "label": LABELS[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "predictions": {LABELS[0]: round(probs[0].item(),4), LABELS[1]: round(probs[1].item(),4)},
    }
