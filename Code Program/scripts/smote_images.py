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
    max_lambda: float = 0.3,
    max_dist_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (train_df + sintetis, tabel_sumber_sintetis).

    Kelas minoritas dinaikkan sampai sama dengan kelas terbesar.

    Pembatas agar tidak muncul "bayangan ganda":
      max_lambda          : campuran maksimal ke induk B (0.3 = hasil tetap didominasi induk A).
      max_dist_percentile : hanya pasangan dengan jarak piksel <= persentil ini (dari semua
                            pasangan tetangga di kelompoknya) yang boleh dicampur; 100 = tanpa
                            penyaringan.
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
        # kumpulkan pasangan (a, b, jarak) per kelompok, lalu saring pasangan yang terlalu jauh
        pair_a: list[int] = []
        pair_b: list[int] = []
        pair_group: list[str] = []
        for g, idx in usable.items():
            k = min(k_neighbors, len(idx) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(vectors[idx])
            dist, neigh = nn.kneighbors(vectors[idx])
            dist, neigh = dist[:, 1:], neigh[:, 1:]
            limit = np.percentile(dist, max_dist_percentile)
            for a in range(len(idx)):
                for col in range(neigh.shape[1]):
                    if dist[a, col] <= limit:
                        pair_a.append(int(idx[a]))
                        pair_b.append(int(idx[neigh[a, col]]))
                        pair_group.append(g)
        if not pair_a:
            raise RuntimeError(f"{class_name}: tidak ada pasangan tetangga yang lolos penyaringan jarak.")

        for n in range(need):
            p = int(rng.integers(len(pair_a)))
            a, b, g = pair_a[p], pair_b[p], pair_group[p]
            lam = float(rng.random()) * max_lambda
            vec = vectors[a] + lam * (vectors[b] - vectors[a])
            array = np.clip(vec.reshape(image_size, image_size) * 255.0, 0, 255).astype(np.uint8)
            path = class_dir / f"smote_{class_name}_{n:04d}.png"
            Image.fromarray(array, mode="L").save(path)
            parent_a = class_frame.loc[a, "image_path"]
            parent_b = class_frame.loc[b, "image_path"]
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
