"""Jumlah gambar per kelas penyakit untuk tiap dataset di project.

Dataset yang dihitung (yang dipakai penelitian):
  1. BRISC2025 (training/validasi/test): brisc2025-v6/.../classification_task/{train,test}
  2. Mendeley (uji eksternal)          : external_datasets/mendeley_82mtzd8x72_v1/raw
Dataset lama (Epic & CSCR) sudah diarsipkan di arsip_dataset_lama/ dan tidak dihitung.

Jalankan (dari folder Code Program):
  python scripts/distribusi_kelas.py
Output: results/class_distribution/{distribusi_kelas.csv, distribusi_kelas.png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/class_distribution"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
COLORS = {"glioma": "#2196F3", "meningioma": "#F44336", "pituitary": "#FF9800", "no_tumor": "#4CAF50"}
EXT = {".jpg", ".jpeg", ".png"}

MENDELEY = ROOT / "external_datasets/mendeley_82mtzd8x72_v1/raw/Brain Cancer MRI Dataset/Brain Cancer MRI Dataset"
# nama folder Mendeley -> nama kelas project
MENDELEY_FOLDERS = {"glioma": "Glioma", "meningioma": "Meningioma",
                    "pituitary": "Pituitary Macroadenoma", "no_tumor": "Healthy"}

DATASETS = {
    "BRISC2025 (dipakai notebook)": (ROOT / "brisc2025-v6/brisc2025/classification_task", ["train", "test"]),
    "Mendeley (dataset eksternal)": (MENDELEY, ["eksternal"]),
}


def count(base: Path, split: str, cls: str) -> int:
    if base == MENDELEY:  # tanpa split; nama folder kelas berbeda
        folder = base / MENDELEY_FOLDERS[cls]
    else:
        folder = base / split / cls
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in EXT) if folder.is_dir() else 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    for name, (base, splits) in DATASETS.items():
        df = pd.DataFrame({s: {c: count(base, s, c) for c in CLASSES} for s in splits})
        df["total"] = df.sum(axis=1)
        df.loc["TOTAL"] = df.sum()
        df["persen_total"] = (df["total"] / df.loc["TOTAL", "total"] * 100).round(1)
        tables[name] = df
        print(f"\n{name}\n{df.to_string()}")

    pd.concat(tables, names=["dataset", "kelas"]).to_csv(OUT / "distribusi_kelas.csv", encoding="utf-8")

    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 5.2), layout="constrained")
    for ax, (name, df) in zip(axes, tables.items()):
        splits = [c for c in df.columns if c not in ("total", "persen_total")]
        width = 0.8 / len(splits)
        for i, split in enumerate(splits):
            xs = [k + (i - (len(splits) - 1) / 2) * width for k in range(len(CLASSES))]
            vals = [df.loc[c, split] for c in CLASSES]
            bars = ax.bar(xs, vals, width * 0.95, label=split, alpha=0.55 + 0.45 * (i == 0),
                          color=[COLORS[c] for c in CLASSES], edgecolor="black", linewidth=0.6,
                          hatch=["", "//", ".."][i])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([f"{c}\n(total {df.loc[c, 'total']}, {df.loc[c, 'persen_total']}%)" for c in CLASSES],
                           fontsize=9)
        ax.set_ylabel("Jumlah gambar")
        ax.set_title(f"{name}\nTotal {df.loc['TOTAL', 'total']} gambar", fontsize=11)
        handles = [plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="black", hatch=["", "//", ".."][i])
                   for i in range(len(splits))]
        ax.legend(handles, splits, title="Split")
    fig.suptitle("Distribusi jumlah gambar per kelas penyakit", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "distribusi_kelas.png", dpi=150)
    print(f"\nDisimpan di {OUT}")


if __name__ == "__main__":
    main()
