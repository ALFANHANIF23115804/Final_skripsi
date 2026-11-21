"""Grad-CAM asli (sama seperti results/per_class_evaluation) + garis hijau area tumor.

Tidak mengubah cara Grad-CAM dihitung: sampel, layer (Conv2d terakhir), ukuran
heatmap, dan judul panel sama dengan results/per_class_evaluation/gradcam.
Satu-satunya tambahan adalah kontur hijau dari mask segmentasi BRISC.

Jalankan (dari folder Code Program):
  python scripts/gradcam_asli_garis_hijau.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_per_class as epc  # noqa: E402


def gradcam_original(model, original, transform, device):
    """Persis seperti evaluate_per_class versi awal: Conv2d terakhir, CAM di-resize ke citra asli."""
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    layer_name, target = layers[-1]
    acts, grads = {}, {}
    fh = target.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    bh = target.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("v", go[0] if not isinstance(go, torch.Tensor) else go)
    )
    try:
        tensor = transform(original).unsqueeze(0).to(device)
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            pred = int(torch.softmax(logits, dim=1)[0].argmax())
            logits[0, pred].backward()
        a, g = acts["v"][0].detach(), grads["v"][0].detach()
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(dim=0, keepdim=True))[None]
        cam = torch.nn.functional.interpolate(
            cam, size=(original.height, original.width), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, layer_name
    finally:
        fh.remove()
        bh.remove()
        model.zero_grad(set_to_none=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "results/per_class_evaluation/gradcam/gradcam_index.csv")
    # Checkpoint yang dipakai results/per_class_evaluation (jangan diganti agar heatmap identik)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "hasil_edgenext/20261104_195431/01_baseline/best_model.pt")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/per_class_evaluation_tta/gradcam")
    ap.add_argument("--mask-dir", type=Path,
                    default=ROOT / "brisc2025-v6/brisc2025/segmentation_task/test/masks")
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _classes, _info = epc.load_checkpoint(ckpt, device)
    transform = epc.make_eval_transform(int(config.get("image_size", 224)))
    index = pd.read_csv(args.index)
    index = index[index["status"] == "created"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in index.iterrows():
        with Image.open(row["image_path"]) as src:
            original = src.convert("RGB")
        cam, layer = gradcam_original(model, original, transform, device)
        mask_path = args.mask_dir / f"{Path(row['image_path']).stem}.png"
        mask = None
        if mask_path.is_file():
            with Image.open(mask_path) as m:
                m = m.convert("L")
                if m.size != original.size:
                    m = m.resize(original.size, Image.Resampling.NEAREST)
                mask = np.asarray(m) > 127

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        axes[0].imshow(original)
        axes[0].set_title(f"Original\nTrue: {row['true_label']}\n"
                          f"Pred: {row['predicted_label']} ({row['confidence']:.3f})")
        axes[1].imshow(original)
        heat = axes[1].imshow(cam, cmap="inferno", alpha=0.9, vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM\nLayer: {layer}")
        axes[2].imshow(original)
        axes[2].imshow(cam, cmap="inferno", alpha=np.clip(cam * 0.75, 0, 0.75), vmin=0, vmax=1)
        axes[2].set_title("Overlay" + (" + area tumor (garis hijau)" if mask is not None else ""))
        if mask is not None:
            for ax in axes:
                ax.contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.6)
        for ax in axes:
            ax.axis("off")
        fig.colorbar(heat, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="relative activation")
        fig.tight_layout()
        out = args.output_dir / Path(row["output_path"]).name
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(("OK  " if mask is not None else "OK (tanpa mask) ") + out.name)

    index.drop(columns=["output_path"]).assign(
        output_path=[str(args.output_dir / Path(p).name) for p in index["output_path"]]
    ).to_csv(args.output_dir / "gradcam_index.csv", index=False)


if __name__ == "__main__":
    main()
