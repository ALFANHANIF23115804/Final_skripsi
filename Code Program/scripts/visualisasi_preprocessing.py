"""Alur preprocessing 4 kelas (glioma, meningioma, pituitary, no_tumor) -> input training.

Tiap baris = satu kelas. Kolom = tahap preprocessing yang sama persis dengan
notebook/evaluate_per_class.py, lalu augmentasi (hanya saat training), lalu
tensor yang masuk ke model.

Jalankan (dari folder Code Program):
  python scripts/visualisasi_preprocessing.py
Output: results/preprocessing/alur_preprocessing_4_kelas.png (+ gambar per kelas)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_per_class as epc  # noqa: E402

TRAIN = ROOT / "brisc2025-v6/brisc2025/classification_task/train"
OUT = ROOT / "results/preprocessing"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
SIZE = 224
SEED = 7


def pick(cls: str) -> Path:
    files = sorted((TRAIN / cls).glob("*.jpg"))
    return files[len(files) // 3]  # satu contoh tetap (reproducible)


def augment(img: Image.Image) -> Image.Image:
    """Contoh augmentasi training (parameter sama dengan Cell 2 notebook)."""
    torch.manual_seed(SEED)
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.25),
    ])
    return tf(img)


def steps(path: Path):
    original = Image.open(path).convert("RGB")
    box = epc.brain_crop_box(original)
    cropped = original.crop(box) if box else original
    padded = epc.crop_brain_pad_square(original)
    resized = padded.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    normalized = epc.preprocess_brain_image(original, SIZE)
    aug = augment(normalized)
    tensor = epc.make_eval_transform(SIZE)(original)
    return original, box, cropped, padded, resized, normalized, aug, tensor


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    titles = ["1. Citra asli", "2. Crop area otak", "3. Pad jadi persegi",
              f"4. Resize {SIZE}x{SIZE}", "5. Normalisasi\npersentil 1-99",
              "6. Augmentasi\n(hanya training)", "7. Tensor input\n(ImageNet norm)"]
    fig, axes = plt.subplots(len(CLASSES), len(titles), figsize=(2.6 * len(titles), 2.8 * len(CLASSES)),
                             layout="constrained")
    for r, cls in enumerate(CLASSES):
        path = pick(cls)
        original, box, cropped, padded, resized, normalized, aug, tensor = steps(path)
        # tensor -> tampilan (min-max per gambar, hanya untuk visualisasi)
        t = torch.as_tensor(tensor)[0].numpy()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        panels = [original, cropped, padded, resized, normalized, aug, t]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if c == 6 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0 and box:
                ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       fill=False, ec="lime", lw=1.6))
            if c == 0:
                ax.set_ylabel(f"{cls}\n{panel.size[0]}x{panel.size[1]}", fontsize=11, fontweight="bold")
            elif c in (1, 2):
                ax.set_xlabel(f"{panel.size[0]}x{panel.size[1]}", fontsize=8)
        # gambar per kelas
        f2, a2 = plt.subplots(1, 6, figsize=(15, 2.9), layout="constrained")
        for c, (panel, ttl) in enumerate(zip(panels[:6], titles[:6])):
            a2[c].imshow(panel); a2[c].set_title(ttl, fontsize=9); a2[c].axis("off")
        f2.suptitle(f"Preprocessing kelas {cls}  ({path.name})")
        f2.savefig(OUT / f"preprocessing_{cls}.png", dpi=150)
        plt.close(f2)
        print(f"OK {cls}: {path.name}")
    fig.suptitle("Alur preprocessing 4 kelas -> hasil dipakai pada tahap training", fontsize=14)
    fig.savefig(OUT / "alur_preprocessing_4_kelas.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
