"""SMOTE untuk citra MRI (hanya dipakai pada data TRAIN).

Cara kerja (SMOTE klasik, ditulis dengan numpy + scikit-learn):
  1. Tiap gambar train dipreproses seperti saat training (crop otak, pad persegi,
     resize 224, normalisasi persentil) lalu diratakan menjadi vektor piksel.
  2. Untuk kelas minoritas: pilih gambar x_i secara acak, ambil satu dari k tetangga
     terdekatnya x_j (kelas SAMA, dan bidang irisan SAMA jika same_plane=True),
     lalu buat gambar baru  x_baru = x_i + lambda * (x_j - x_i),  lambda ~ U(0, 1).
  3. Gambar sintetis disimpan sebagai PNG dan ditambahkan ke manifest train.

Catatan penting:
  - Jangan pernah diterapkan ke validation/test (akan membocorkan data dan menipu metrik).
  - Interpolasi piksel dua MRI yang tidak sejajar menghasilkan gambar "bayangan ganda".
    Pembatasan same_plane + tetangga terdekat mengurangi, bukan menghilangkan, efek itu.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def _vector(path: str, preprocess_fn: Callable, image_size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = preprocess_fn(source.convert("RGB"), image_size)
    return np.asarray(image.convert("L"), dtype=np.float32).ravel() / 255.0


def smote_oversample(
    train_df: pd.DataFrame,
    preprocess_fn: Callable,
    output_dir: str | Path,
    class_names: list[str],
    image_size: int = 224,
    k_neighbors: int = 5,
    same_plane: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (train_df + sintetis, tabel_sumber_sintetis).

    Kelas minoritas dinaikkan sampai sama dengan kelas terbesar.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    counts = train_df["label"].value_counts()
    target = int(counts.max())

    new_rows: list[dict] = []
    sources: list[dict] = []
    for class_name in class_names:
        class_frame = train_df[train_df["label"] == class_name].reset_index(drop=True)
        need = target - len(class_frame)
        if need <= 0:
            continue
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        vectors = np.stack([_vector(p, preprocess_fn, image_size) for p in class_frame["image_path"]])

        # kelompok tetangga: per bidang irisan (atau semua jika same_plane=False)
        groups = (
            {plane: idx.to_numpy() for plane, idx in class_frame.groupby("plane").groups.items()}
            if same_plane
            else {"semua": np.arange(len(class_frame))}
        )
        usable = {g: idx for g, idx in groups.items() if len(idx) >= 2}
        if not usable:
            raise RuntimeError(f"{class_name}: tidak ada kelompok dengan >= 2 gambar untuk SMOTE.")
        neighbours: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for g, idx in usable.items():
            k = min(k_neighbors, len(idx) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(vectors[idx])
            neighbours[g] = (idx, nn.kneighbors(vectors[idx], return_distance=False)[:, 1:])

        # sampel dasar dipilih sebanding ukuran kelompok
        group_names = list(usable)
        weights = np.array([len(usable[g]) for g in group_names], dtype=float)
        weights /= weights.sum()
        for n in range(need):
            g = group_names[rng.choice(len(group_names), p=weights)]
            idx, neigh = neighbours[g]
            a = int(rng.integers(len(idx)))
            b = int(neigh[a][rng.integers(neigh.shape[1])])
            lam = float(rng.random())
            vec = vectors[idx[a]] + lam * (vectors[idx[b]] - vectors[idx[a]])
            array = np.clip(vec.reshape(image_size, image_size) * 255.0, 0, 255).astype(np.uint8)
            path = class_dir / f"smote_{class_name}_{n:04d}.png"
            Image.fromarray(array, mode="L").save(path)
            parent_a = class_frame.loc[idx[a], "image_path"]
            parent_b = class_frame.loc[idx[b], "image_path"]
            new_rows.append({
                "image_path": str(path.resolve()),
                "split": "train",
                "label": class_name,
                "label_id": class_names.index(class_name),
                "filename": path.name,
                "plane": g if g != "semua" else "unknown",
                "file_sha256": "",
                "synthetic": True,
            })
            sources.append({"synthetic": path.name, "label": class_name, "parent_a": parent_a,
                            "parent_b": parent_b, "lambda": round(lam, 4)})

    base = train_df.copy()
    base["synthetic"] = False
    if not new_rows:
        return base, pd.DataFrame(columns=["synthetic", "label", "parent_a", "parent_b", "lambda"])
    combined = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
    return combined, pd.DataFrame(sources)
