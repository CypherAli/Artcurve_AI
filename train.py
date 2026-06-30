import torch, torch.nn as nn, time
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

DATA_DIR = "/home/coder/artcurve-ai/data/binary"
MODEL_DIR = "/home/coder/artcurve-ai/model"

device = torch.device("cuda")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),
])

print("Loading dataset...")
ds = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"Classes: {ds.classes}, Total: {len(ds)}")
tr, va = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
tl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
vl = DataLoader(va, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

print("Loading CLIP...")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
vision = get_peft_model(clip.vision_model, LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"],lora_dropout=0.1))

class D(nn.Module):
    def __init__(s,v):
        super().__init__()
        s.v=v; s.h=nn.Sequential(nn.Linear(768,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,2))
    def forward(s,x): return s.h(s.v(pixel_values=x).pooler_output)

model = D(vision).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")
opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=2e-4)
crit = nn.CrossEntropyLoss()

for ep in range(3):
    model.train(); ls,co,n=0,0,0; t0=time.time()
    for i,(x,y) in enumerate(tl):
        x,y=x.to(device),y.to(device); o=model(x); l=crit(o,y)
        opt.zero_grad(); l.backward(); opt.step()
        ls+=l.item(); co+=(o.argmax(1)==y).sum().item(); n+=y.size(0)
        if (i+1)%200==0: print(f"  E{ep+1} [{i+1}/{len(tl)}] loss={ls/(i+1):.4f} acc={100*co/n:.1f}%",flush=True)
    model.eval(); vc,vn=0,0
    with torch.no_grad():
        for x,y in vl: x,y=x.to(device),y.to(device); vc+=(model(x).argmax(1)==y).sum().item(); vn+=y.size(0)
    va_acc=100*vc/vn
    print(f"Epoch {ep+1}/3: train={100*co/n:.1f}% val={va_acc:.1f}% time={time.time()-t0:.0f}s",flush=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pt")
print("Done!")
