import torch, torch.nn as nn, glob, random
from torchvision import transforms
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from PIL import Image

device = torch.device("cuda")

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
model.load_state_dict(torch.load("/home/coder/artcurve-ai/model/best_model.pt"))
model.eval()

LABELS = ["AI", "REAL"]  # alphabetical: ai=0, real=1

# Test 20 random images from each class — NOT used in training necessarily but quick sanity check
ai_files = glob.glob("/home/coder/artcurve-ai/data/binary/ai/*")
real_files = glob.glob("/home/coder/artcurve-ai/data/binary/real/*")
random.seed(42)
test_files = [(f, "AI") for f in random.sample(ai_files, 15)] + [(f, "REAL") for f in random.sample(real_files, 15)]
random.shuffle(test_files)

correct = 0
for path, true_label in test_files:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = LABELS[probs.argmax().item()]
    conf = probs.max().item()
    ok = "✓" if pred == true_label else "✗"
    if pred == true_label: correct += 1
    print(f"{ok} true={true_label:5s} pred={pred:5s} conf={conf:.2f}  {path.split('/')[-1][:50]}")

print(f"\nAccuracy on 30 sample check: {correct}/30 = {100*correct/30:.1f}%")
