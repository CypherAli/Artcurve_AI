import os, shutil
from pathlib import Path

SRC = Path("/home/coder/.cache/kagglehub/datasets/ravidussilva/real-ai-art/versions/5/Real_AI_SD_LD_Dataset/train")
DST = Path("/home/coder/artcurve-ai/data/binary")

(DST / "real").mkdir(parents=True, exist_ok=True)
(DST / "ai").mkdir(parents=True, exist_ok=True)

real_count = 0
ai_count = 0

for folder in sorted(SRC.iterdir()):
    if not folder.is_dir():
        continue
    name = folder.name
    is_ai = name.startswith("AI_")
    target = DST / ("ai" if is_ai else "real")
    
    for img in folder.iterdir():
        if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            dst_file = target / f"{name}_{img.name}"
            if not dst_file.exists():
                os.symlink(img, dst_file)
            if is_ai:
                ai_count += 1
            else:
                real_count += 1

print(f"Real: {real_count}, AI: {ai_count}")
print(f"Symlinked to {DST}")
