"""Detailed, leakage-aware evaluation for the existing EdgeNeXt checkpoint.

This module intentionally performs inference only.  It does not train, alter,
or overwrite the existing notebook/checkpoints.  By default it evaluates the
newest saved baseline checkpoint on the official ``test`` split.

Run from this directory with::

    python evaluate_per_class.py

Use ``--checkpoint`` to evaluate another saved checkpoint, for example the
tuned trial or a future experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError as error:  # pragma: no cover - message is handled by main()
    timm = None
    TIMM_IMPORT_ERROR = error
else:
    TIMM_IMPORT_ERROR = None


ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def json_default(value: Any) -> Any:
    """Convert common numpy/PyTorch/path values to JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def resolve_data_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "brisc2025-v6" / "brisc2025" / "classification_task",
        project_root.parent / "brisc2025-v6" / "brisc2025" / "classification_task",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate.resolve()
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Dataset classification_task tidak ditemukan. Lokasi yang diperiksa:\n" + checked
    )


def find_default_checkpoint(project_root: Path) -> Path:
    candidates = list((project_root / "hasil_edgenext").glob("*/01_baseline/best_model.pt"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "Checkpoint baseline tidak ditemukan di hasil_edgenext/*/01_baseline/best_model.pt. "
            "Gunakan --checkpoint untuk menunjuk checkpoint yang tersedia."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_plane(filename: str) -> str:
    name = filename.lower()
    tokens = set(re.split(r"[_\-.]+", name))
    if "axial" in name or "ax" in tokens:
        return "axial"
    if "coronal" in name or "co" in tokens:
        return "coronal"
    if "sagittal" in name or "sa" in tokens:
        return "sagittal"
    return "unknown"


def scan_split(data_dir: Path, split: str, class_names: list[str]) -> pd.DataFrame:
    """Read one split without changing the files or the split assignment."""

    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Folder split tidak ditemukan: {split_dir}")

    records: list[dict[str, Any]] = []
    for label_id, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas {class_name!r} tidak ditemukan: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved = image_path.resolve()
                records.append(
                    {
                        "image_path": str(resolved),
                        "split": split,
                        "true_label": class_name,
                        "label_id": label_id,
                        "filename": resolved.name,
                        "plane": detect_plane(resolved.name),
                        "file_sha256": file_sha256(resolved),
                    }
                )
    if not records:
        raise RuntimeError(f"Tidak ada gambar valid di {split_dir}")
    frame = pd.DataFrame(records)
    return frame.reset_index(drop=True)


def find_manifest(project_root: Path, data_dir: Path) -> Path | None:
    candidates = [
        data_dir.parent / "manifest.csv",
        data_dir.parent.parent / "manifest.csv",
        project_root / "brisc2025-v6" / "brisc2025" / "manifest.csv",
        project_root.parent / "brisc2025-v6" / "brisc2025" / "manifest.csv",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def audit_data_leakage(
    project_root: Path, data_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Audit exact image overlap and report whether patient/group IDs exist."""

    train_hashes = set(train_df["file_sha256"])
    test_hashes = set(test_df["file_sha256"])
    exact_overlap = sorted(train_hashes.intersection(test_hashes))
    effective_train_df = train_df[~train_df["file_sha256"].isin(test_hashes)]
    deduplicated_train_df = effective_train_df.drop_duplicates("file_sha256", keep="first")
    effective_overlap = sorted(
        set(effective_train_df["file_sha256"]).intersection(test_hashes)
    )

    manifest_path = find_manifest(project_root, data_dir)
    patient_columns: list[str] = []
    patient_overlap: str = "not_checked"
    manifest_note = "manifest tidak ditemukan"
    if manifest_path is not None:
        try:
            manifest = pd.read_csv(manifest_path)
            normalized = {str(column).lower().strip(): column for column in manifest.columns}
            patient_columns = [
                str(normalized[key])
                for key in normalized
                if any(token in key for token in ("patient", "subject", "group", "study"))
            ]
            if patient_columns:
                split_column = next(
                    (normalized[key] for key in normalized if key in {"split", "subset", "partition"}),
                    None,
                )
                if split_column is not None:
                    values: set[str] = set()
                    for column in patient_columns:
                        train_ids = set(
                            manifest.loc[
                                manifest[split_column].astype(str).str.lower().eq("train"), column
                            ].dropna().astype(str)
                        )
                        test_ids = set(
                            manifest.loc[
                                manifest[split_column].astype(str).str.lower().eq("test"), column
                            ].dropna().astype(str)
                        )
                        if train_ids.intersection(test_ids):
                            values.update(train_ids.intersection(test_ids))
                    patient_overlap = str(len(values))
                else:
                    patient_overlap = "manifest_has_id_but_no_split_column"
                manifest_note = "patient/group identifier ditemukan; overlap diaudit dari manifest"
            else:
                patient_overlap = "not_available"
                manifest_note = "manifest tidak memiliki patient_id/subject_id/group_id/study_id"
        except Exception as error:
            patient_overlap = f"audit_error: {error}"
            manifest_note = "manifest gagal dibaca"
    else:
        patient_overlap = "not_available"

    rows = [
        {"check": "dataset_dir", "value": str(data_dir), "status": "observed"},
        {"check": "train_images", "value": len(train_df), "status": "observed"},
        {"check": "test_images", "value": len(test_df), "status": "observed"},
        {
            "check": "exact_sha256_overlap_train_test",
            "value": len(exact_overlap),
            "status": "raw_dataset_observation",
        },
        {
            "check": "exact_sha256_overlap_effective_training_test",
            "value": len(effective_overlap),
            "status": "pass" if not effective_overlap else "warning",
        },
        {
            "check": "training_rows_removed_by_test_hash_deduplication",
            "value": int(len(train_df) - len(effective_train_df)),
            "status": "not_used_for_training" if exact_overlap else "not_needed",
        },
        {
            "check": "training_rows_removed_by_internal_hash_deduplication",
            "value": int(len(effective_train_df) - len(deduplicated_train_df)),
            "status": "not_used_for_training",
        },
        {
            "check": "training_rows_after_notebook_hash_cleanup",
            "value": int(len(deduplicated_train_df)),
            "status": "effective_training_input_reference",
        },
        {
            "check": "patient_or_group_columns",
            "value": ", ".join(patient_columns) if patient_columns else "not_available",
            "status": "observed",
        },
        {
            "check": "patient_or_group_overlap_train_test",
            "value": patient_overlap,
            "status": "pass" if patient_overlap in {"0", "not_checked", "not_available"} else "warning",
        },
        {"check": "manifest", "value": str(manifest_path) if manifest_path else "not_found", "status": "observed"},
        {"check": "leakage_note", "value": manifest_note, "status": "note"},
    ]
    return pd.DataFrame(rows)


def brain_crop_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    low, high = np.percentile(gray, [5, 99])
    threshold = max(3.0, low + 0.12 * (high - low))
    foreground = gray > threshold
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if len(rows) and len(cols):
        return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return None


def mask_to_model_view(image: Image.Image, mask: Image.Image, image_size: int) -> np.ndarray:
    """Terapkan crop/pad/resize yang sama seperti citra ke mask segmentasi."""

    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    box = brain_crop_box(image)
    if box is not None:
        mask = mask.crop(box)
    side = max(mask.size)
    left = (side - mask.width) // 2
    top = (side - mask.height) // 2
    mask = ImageOps.expand(
        mask, border=(left, top, side - mask.width - left, side - mask.height - top), fill=0
    ).resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def crop_brain_pad_square(image: Image.Image) -> Image.Image:
    """Match the notebook's crop/pad preprocessing exactly."""

    box = brain_crop_box(image)
    if box is not None:
        image = image.crop(box)
    side = max(image.size)
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    return ImageOps.expand(
        image,
        border=(left, top, side - image.width - left, side - image.height - top),
        fill=0,
    ).convert("RGB")


def preprocess_brain_image(image: Image.Image, image_size: int) -> Image.Image:
    image = crop_brain_pad_square(image).resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    array = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = np.clip(
            (array - low) / (high - low) * 255.0, 0, 255
        ).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: preprocess_brain_image(image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class BrainTumorTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose, image_size: int):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.image_size = int(image_size)
        missing = [path for path in self.frame["image_path"] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} file test tidak ditemukan; contoh: {missing[:3]}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        path = Path(row["image_path"])
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Gagal membaca gambar {path}: {error}") from error
        tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Transform gambar harus menghasilkan torch.Tensor")
        expected = (3, self.image_size, self.image_size)
        if tuple(tensor.shape) != expected:
            raise RuntimeError(f"Preprocessing menghasilkan {tuple(tensor.shape)}, expected {expected}")
        return tensor, int(row["label_id"]), str(path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], list[str], dict[str, Any]]:
    if timm is None:
        raise ImportError(f"timm belum tersedia: {TIMM_IMPORT_ERROR}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Format checkpoint tidak dikenali; diperlukan dictionary PyTorch.")
    config = checkpoint.get("config", {}) or {}
    class_names = list(checkpoint.get("class_names", []))
    if not class_names:
        raise ValueError("Checkpoint tidak menyimpan class_names; evaluasi dihentikan agar label tidak diasumsikan.")
    model_name = checkpoint.get("model_name") or config.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint tidak menyimpan model_name.")
    drop_rate = float(config.get("drop_rate", 0.0))
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(class_names),
            drop_rate=drop_rate,
        )
    except TypeError:
        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint tidak memiliki model_state_dict/state_dict.")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint tidak cocok dengan arsitektur {model_name}. Missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.to(device)
    model.eval()
    info = {
        "checkpoint": checkpoint_path,
        "model_name": model_name,
        "config": config,
        "class_names": class_names,
        "device": device,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    return model, config, class_names, info


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, use_tta: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []
    all_paths: list[str] = []
    start = time.perf_counter()
    for images, targets, paths in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        probabilities = torch.softmax(model(images), dim=1)
        if use_tta:  # rata-rata citra asli + flip horizontal
            probabilities = (probabilities + torch.softmax(model(torch.flip(images, dims=[3])), dim=1)) / 2
        predictions = probabilities.argmax(dim=1)
        all_true.extend(targets.numpy().tolist())
        all_pred.extend(predictions.cpu().numpy().tolist())
        all_prob.append(probabilities.cpu().numpy())
        all_paths.extend(paths)
    elapsed = time.perf_counter() - start
    return (
        np.asarray(all_true, dtype=int),
        np.asarray(all_pred, dtype=int),
        np.concatenate(all_prob, axis=0),
        all_paths,
        elapsed,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    if not isinstance(class_report, dict):
        raise TypeError("classification_report harus menghasilkan dictionary")
    class_rows: list[dict[str, Any]] = []
    total = int(cm.sum())
    for class_id, class_name in enumerate(class_names):
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id, :].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = int(total - tp - fn - fp)
        binary_true = y_true == class_id
        roc_auc = float(roc_auc_score(binary_true, y_prob[:, class_id])) if np.unique(binary_true).size == 2 else float("nan")
        pr_auc = float(average_precision_score(binary_true, y_prob[:, class_id])) if np.unique(binary_true).size == 2 else float("nan")
        row = class_report[class_name]
        if not isinstance(row, dict):
            raise TypeError(f"Metrik kelas {class_name!r} harus berupa dictionary")
        class_rows.append(
            {
                "class": class_name,
                "precision": float(row["precision"]),
                "recall_sensitivity": float(row["recall"]),
                "specificity": safe_divide(tn, tn + fp),
                "f1_score": float(row["f1-score"]),
                "support": int(row["support"]),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "roc_auc_ovr": roc_auc,
                "pr_auc_ovr": pr_auc,
            }
        )
    per_class = pd.DataFrame(class_rows)
    macro = per_class[["precision", "recall_sensitivity", "specificity", "f1_score"]].mean()
    weights = per_class["support"].to_numpy(dtype=float)
    weighted = {
        metric: float(np.average(per_class[metric], weights=weights))
        for metric in ["precision", "recall_sensitivity", "specificity", "f1_score"]
    }
    report_rows = per_class[["class", "precision", "recall_sensitivity", "specificity", "f1_score", "support"]].copy()
    report_rows = pd.concat(
        [
            report_rows,
            pd.DataFrame(
                [
                    {
                        "class": "accuracy",
                        "precision": float("nan"),
                        "recall_sensitivity": float("nan"),
                        "specificity": float("nan"),
                        "f1_score": float(accuracy_score(y_true, y_pred)),
                        "support": total,
                    },
                    {
                        "class": "macro avg",
                        "precision": float(macro["precision"]),
                        "recall_sensitivity": float(macro["recall_sensitivity"]),
                        "specificity": float(macro["specificity"]),
                        "f1_score": float(macro["f1_score"]),
                        "support": total,
                    },
                    {
                        "class": "weighted avg",
                        "precision": weighted["precision"],
                        "recall_sensitivity": weighted["recall_sensitivity"],
                        "specificity": weighted["specificity"],
                        "f1_score": weighted["f1_score"],
                        "support": total,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(macro["precision"]),
        "recall_macro": float(macro["recall_sensitivity"]),
        "specificity_macro": float(macro["specificity"]),
        "macro_f1": float(macro["f1_score"]),
        "precision_weighted": weighted["precision"],
        "recall_weighted": weighted["recall_sensitivity"],
        "specificity_weighted": weighted["specificity"],
        "weighted_f1": weighted["f1_score"],
        "roc_auc_ovr_macro": float(per_class["roc_auc_ovr"].mean()),
        "pr_auc_ovr_macro": float(per_class["pr_auc_ovr"].mean()),
        "support_total": total,
    }
    return per_class, report_rows, overall, cm


def save_confusion_matrices(cm: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    for normalized, filename, title, fmt in [
        (False, "confusion_matrix.png", "Confusion matrix", "d"),
        (True, "normalized_confusion_matrix.png", "Normalized confusion matrix", ".2f"),
    ]:
        values = cm.astype(float)
        if normalized:
            row_sums = values.sum(axis=1, keepdims=True)
            values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
        else:
            values = cm
        plt.figure(figsize=(8.5, 7))
        sns.heatmap(
            values,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0 if normalized else None,
            vmax=1 if normalized else None,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def save_metric_plot(per_class: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = ["precision", "recall_sensitivity", "specificity", "f1_score"]
    labels = {
        "precision": "Precision",
        "recall_sensitivity": "Recall / sensitivity",
        "specificity": "Specificity",
        "f1_score": "F1-score",
    }
    figure, axis = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(per_class))
    width = 0.19
    for offset, column in enumerate(metric_columns):
        axis.bar(x + (offset - 1.5) * width, per_class[column], width, label=labels[column])
    axis.set_xticks(x, per_class["class"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Per-class evaluation metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "per_class_metrics.png", dpi=180)
    plt.close(figure)


def save_curve_plots(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], output_dir: Path
) -> None:
    labels = list(range(len(class_names)))
    figure, axis = plt.subplots(figsize=(9, 7))
    for class_id, class_name in zip(labels, class_names):
        binary_true = y_true == class_id
        if np.unique(binary_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, class_id])
        auc = roc_auc_score(binary_true, y_prob[:, class_id])
        axis.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={auc:.4f})")
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("One-vs-rest ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    for class_id, class_name in zip(labels, class_names):
        binary_true = y_true == class_id
        if np.unique(binary_true).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, y_prob[:, class_id])
        ap = average_precision_score(binary_true, y_prob[:, class_id])
        axis.plot(recall, precision, linewidth=2, label=f"{class_name} (AP={ap:.4f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("One-vs-rest precision-recall curves")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(figure)


def make_prediction_frame(
    paths: list[str], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "image_path": paths,
            "true_label": [class_names[index] for index in y_true],
            "predicted_label": [class_names[index] for index in y_pred],
            "confidence": y_prob.max(axis=1),
            "correct": y_true == y_pred,
            "true_label_id": y_true,
            "predicted_label_id": y_pred,
        }
    )
    for class_id, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = y_prob[:, class_id]
    return frame


def save_image_grid(frame: pd.DataFrame, output_path: Path, title: str, max_items: int = 16) -> None:
    subset = frame.head(max_items).copy()
    if subset.empty:
        return
    columns = min(4, len(subset))
    rows = math.ceil(len(subset) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, (_, row) in zip(axes, subset.iterrows()):
        try:
            with Image.open(row["image_path"]) as source:
                axis.imshow(source.convert("RGB"))
        except Exception as error:
            axis.text(0.5, 0.5, f"Gagal membaca\n{error}", ha="center", va="center")
        axis.set_title(
            f"True: {row['true_label']}\nPred: {row['predicted_label']}\n"
            f"Confidence: {float(row['confidence']):.3f}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[len(subset) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_error_analysis(predictions: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wrong = predictions.loc[~predictions["correct"]].copy()
    wrong = wrong.sort_values(["true_label", "predicted_label", "confidence"])
    wrong.to_csv(output_dir / "misclassified.csv", index=False)
    pairs = (
        wrong.groupby(["true_label", "predicted_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    pairs.to_csv(output_dir / "confusion_pairs.csv", index=False)
    for (true_label, predicted_label), group in wrong.groupby(["true_label", "predicted_label"]):
        safe_name = f"{true_label}_to_{predicted_label}".replace(" ", "_")
        save_image_grid(
            group.sort_values("confidence"),
            output_dir / "misclassified_by_pair" / f"{safe_name}.png",
            f"Misclassified: {true_label} -> {predicted_label}",
            max_items=12,
        )
    save_image_grid(
        wrong.sort_values("confidence"),
        output_dir / "misclassified_examples.png",
        "Misclassified test images (lowest confidence first)",
    )
    low_confidence = predictions.sort_values("confidence").head(20).copy()
    low_confidence.to_csv(output_dir / "low_confidence.csv", index=False)
    save_image_grid(
        low_confidence,
        output_dir / "low_confidence_examples.png",
        "Lowest-confidence test images",
        max_items=16,
    )
    return wrong, pairs, low_confidence


def run_gradcam(
    model: nn.Module,
    row: pd.Series,
    role: str,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
    output_path: Path,
    mask_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a Grad-CAM panel; jika mask tersedia, batas tumor digambar dengan garis hijau."""

    with Image.open(row["image_path"]) as source:
        original = source.convert("RGB")
    cam, layer_name, _ = compute_gradcam(model, original, transform, device)
    model_view = preprocess_brain_image(original, cam.shape[0])
    mask = load_model_view_mask(row["image_path"], original, mask_dir, cam.shape[0])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    axes[0].imshow(model_view)
    axes[0].set_title(
        f"Input model\nTrue: {row['true_label']}\nPred: {row['predicted_label']} ({row['confidence']:.3f})"
    )
    axes[1].imshow(model_view)
    heatmap = axes[1].imshow(cam, cmap="inferno", alpha=0.9, vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM\nLayer: {layer_name}")
    axes[2].imshow(model_view)
    axes[2].imshow(cam, cmap="inferno", alpha=np.clip(cam * 0.75, 0, 0.75), vmin=0, vmax=1)
    axes[2].set_title("Overlay" + (" + mask tumor (hijau)" if mask is not None else ""))
    for axis in axes:
        if mask is not None:
            draw_mask_outline(axis, mask)
        axis.axis("off")
    figure.colorbar(heatmap, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="relative activation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "status": "created",
        "role": role,
        "image_path": row["image_path"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": float(row["confidence"]),
        "layer": layer_name,
        "output_path": output_path,
    }


def compute_gradcam(
    model: nn.Module, original: Image.Image, transform: transforms.Compose, device: torch.device
) -> tuple[np.ndarray, str, int]:
    """Grad-CAM (0..1, ukuran input model) untuk kelas prediksi.

    Target = output stage terakhir (peta fitur penuh sebelum pooling). Conv2d terakhir
    pada EdgeNeXt adalah konvolusi depthwise di dalam blok SDTA yang hanya memegang
    sebagian channel, sehingga tidak mewakili fitur yang dipakai classifier.
    """

    stages = getattr(model, "stages", None)
    if isinstance(stages, nn.Sequential) and len(stages) > 0:
        layer_name, target_layer = f"stages.{len(stages) - 1}", stages[-1]
    else:
        conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("Model tidak memiliki Conv2d; Grad-CAM tidak tersedia.")
        layer_name, target_layer = conv_layers[-1]
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            activations["value"] = output

    def backward_hook(
        _module: nn.Module,
        _grad_input: torch.Tensor | tuple[torch.Tensor | None, ...],
        grad_output: torch.Tensor | tuple[torch.Tensor | None, ...],
    ) -> None:
        if isinstance(grad_output, torch.Tensor):
            gradients["value"] = grad_output
        elif grad_output and isinstance(grad_output[0], torch.Tensor):
            gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        transformed = transform(original)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("Transform gambar harus menghasilkan torch.Tensor")
        tensor = transformed.unsqueeze(0).to(device)
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_id = int(probabilities.argmax().item())
            logits[0, predicted_id].backward()
        if "value" not in activations or "value" not in gradients:
            raise RuntimeError(f"Layer {layer_name} tidak menghasilkan activation/gradient spasial.")
        activation = activations["value"][0].detach()
        gradient = gradients["value"][0].detach()
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0, keepdim=True))[None]
        # CAM disejajarkan dengan input model (setelah crop/pad/resize), bukan citra asli,
        # supaya posisi heatmap sesuai dengan yang benar-benar dilihat model.
        cam = torch.nn.functional.interpolate(
            cam,
            size=tuple(tensor.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam, layer_name, predicted_id
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def load_model_view_mask(
    image_path: str | Path, original: Image.Image, mask_dir: Path | None, image_size: int
) -> np.ndarray | None:
    """Mask segmentasi BRISC yang sudah disejajarkan dengan input model; None jika tidak ada."""

    if mask_dir is None:
        return None
    mask_path = Path(mask_dir) / f"{Path(image_path).stem}.png"
    if not mask_path.is_file():
        return None
    with Image.open(mask_path) as source_mask:
        mask = mask_to_model_view(original, source_mask, image_size)
    return mask if mask.any() else None


def draw_mask_outline(axis: Any, mask: np.ndarray, color: str = "lime") -> None:
    axis.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=1.4)


def save_cam_mask_grid(items: list[dict[str, Any]], output_path: Path, title: str, per_row: int = 6) -> None:
    """Grid 2 baris per kelompok: atas = input + garis mask, bawah = Grad-CAM + garis mask."""

    if not items:
        return
    groups = [items[start : start + per_row] for start in range(0, len(items), per_row)]
    columns = min(per_row, len(items))
    figure, axes = plt.subplots(
        2 * len(groups), columns, figsize=(3.3 * columns, 7.4 * len(groups)), squeeze=False, layout="constrained"
    )
    for group_index, group in enumerate(groups):
        for column in range(columns):
            top, bottom = axes[2 * group_index, column], axes[2 * group_index + 1, column]
            top.axis("off")
            bottom.axis("off")
            if column >= len(group):
                continue
            item = group[column]
            top.imshow(item["view"])
            draw_mask_outline(top, item["mask"])
            top.set_title(
                f"{item['name']}\n{item['true_label']} -> {item['predicted_label']} ({item['confidence']:.2f})\n"
                f"CAM dekat tumor: {item['cam_near_mask']:.0%}",
                fontsize=8,
            )
            bottom.imshow(item["view"])
            bottom.imshow(item["cam"], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            draw_mask_outline(bottom, item["mask"])
    figure.suptitle(title + "\nGaris hijau = mask tumor (BRISC); warna terang = perhatian Grad-CAM", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_gradcam_examples(
    model: nn.Module,
    predictions: pd.DataFrame,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    mask_dir: Path | None = None,
) -> pd.DataFrame:
    selections: list[tuple[str, pd.Series]] = []
    for role, subset in [
        ("correct", predictions[predictions["correct"]].sort_values("confidence").head(3)),
        ("wrong", predictions[~predictions["correct"]].sort_values("confidence").head(3)),
        ("low_confidence", predictions.sort_values("confidence").head(3)),
    ]:
        for _, row in subset.iterrows():
            selections.append((role, row))
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, (role, row) in enumerate(selections, start=1):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row["image_path"]).stem)
        filename = f"{role}_{index:02d}_{stem}.png"
        if filename in used_names:
            filename = f"{role}_{index:02d}_{index}_{stem}.png"
        used_names.add(filename)
        output_path = output_dir / filename
        try:
            records.append(
                run_gradcam(model, row, role, transform, class_names, device, output_path, mask_dir)
            )
        except Exception as error:
            records.append(
                {
                    "status": "error",
                    "role": role,
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": float(row["confidence"]),
                    "error": str(error),
                    "output_path": output_path,
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "gradcam_index.csv", index=False)
    return frame


def save_gradcam_mask_overlap(
    model: nn.Module,
    predictions: pd.DataFrame,
    transform: transforms.Compose,
    device: torch.device,
    mask_dir: Path,
    image_size: int,
    output_dir: Path,
    correct_samples: int = 150,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Ukur berapa bagian perhatian Grad-CAM yang jatuh di dalam mask tumor.

    cam_in_mask   : porsi energi CAM di dalam mask (1.0 = semua perhatian di tumor)
    pointing_hit  : titik CAM tertinggi berada di dalam mask
    mask_area_frac: luas mask di input model (pembanding: nilai cam_in_mask acak ~ luas ini)
    """

    from scipy.ndimage import binary_dilation

    if not mask_dir.is_dir():
        print(f"INFO: folder mask tidak ditemukan ({mask_dir}); metrik Grad-CAM-mask dilewati.")
        return None
    tumor = predictions[predictions["true_label"] != "no_tumor"]
    wrong = tumor[~tumor["correct"]]
    right = tumor[tumor["correct"]]
    right = right.sample(n=min(correct_samples, len(right)), random_state=seed)
    records: list[dict[str, Any]] = []
    grid_wrong: list[dict[str, Any]] = []
    grid_right: list[dict[str, Any]] = []
    for _, row in pd.concat([wrong, right]).iterrows():
        with Image.open(row["image_path"]) as source:
            original = source.convert("RGB")
        mask = load_model_view_mask(row["image_path"], original, mask_dir, image_size)
        if mask is None:
            continue
        cam, _, _ = compute_gradcam(model, original, transform, device)
        peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
        # Grad-CAM stage terakhir beresolusi 7x7, jadi sediakan juga versi toleran (+16 px)
        near = binary_dilation(mask, iterations=16)
        cam_near_mask = float(cam[near].sum() / (cam.sum() + 1e-8))
        target = grid_wrong if not row["correct"] else (grid_right if len(grid_right) < 6 else None)
        if target is not None:
            target.append(
                {
                    "view": preprocess_brain_image(original, image_size),
                    "mask": mask,
                    "cam": cam,
                    "name": re.sub(r"^brisc2025_test_", "", Path(row["image_path"]).stem),
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": float(row["confidence"]),
                    "cam_near_mask": cam_near_mask,
                }
            )
        records.append(
            {
                "image_path": row["image_path"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "correct": bool(row["correct"]),
                "confidence": float(row["confidence"]),
                "cam_in_mask": float(cam[mask].sum() / (cam.sum() + 1e-8)),
                "pointing_hit": bool(mask[peak]),
                "mask_area_frac": float(mask.mean()),
                "cam_near_mask": cam_near_mask,
                "pointing_hit_near": bool(near[peak]),
                "near_area_frac": float(near.mean()),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "gradcam_mask_overlap.csv", index=False)
    grid_wrong.sort(key=lambda item: item["confidence"])
    save_cam_mask_grid(grid_wrong, output_dir / "gradcam_vs_mask_salah.png", "Grad-CAM vs mask tumor - prediksi SALAH")
    save_cam_mask_grid(grid_right, output_dir / "gradcam_vs_mask_benar.png", "Grad-CAM vs mask tumor - contoh prediksi BENAR")
    summary = (
        frame.groupby("correct")
        .agg(
            n=("cam_in_mask", "size"),
            cam_in_mask_median=("cam_in_mask", "median"),
            pointing_hit_rate=("pointing_hit", "mean"),
            mask_area_frac_median=("mask_area_frac", "median"),
            cam_near_mask_median=("cam_near_mask", "median"),
            pointing_hit_near_rate=("pointing_hit_near", "mean"),
            near_area_frac_median=("near_area_frac", "median"),
        )
        .reset_index()
    )
    if frame["correct"].nunique() == 2:
        from scipy.stats import mannwhitneyu

        p_value = mannwhitneyu(
            frame.loc[frame["correct"], "cam_in_mask"],
            frame.loc[~frame["correct"], "cam_in_mask"],
            alternative="two-sided",
        ).pvalue
        summary["mannwhitney_p_cam_in_mask"] = p_value
    summary.to_csv(output_dir / "gradcam_mask_overlap_summary.csv", index=False)
    print("\nGrad-CAM di dalam mask tumor (benar vs salah):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return frame


def save_experiment_plan(output_dir: Path, class_names: list[str], baseline: dict[str, Any], per_class: pd.DataFrame) -> None:
    """Prepare isolated follow-up experiments without launching training."""

    baseline_row: dict[str, Any] = {
        "experiment": "baseline_existing_checkpoint",
        "change": "none",
        "status": "completed",
        "macro_f1": baseline["macro_f1"],
        "balanced_accuracy": baseline["balanced_accuracy"],
    }
    for _, row in per_class.iterrows():
        baseline_row[f"recall_{row['class']}"] = row["recall_sensitivity"]
    rows = [baseline_row]
    variants = [
        ("class_weight", "CrossEntropyLoss(class weights only)"),
        ("focal_loss", "Focal loss only"),
        ("data_augmentation", "Augmentation policy only"),
        ("minority_oversampling", "WeightedRandomSampler only"),
        ("clahe_contrast", "CLAHE/contrast preprocessing only"),
        ("image_size", "Image size only, e.g. 256 or 288"),
    ]
    for experiment, change in variants:
        row = {"experiment": experiment, "change": change, "status": "not_run"}
        row.update({"macro_f1": "", "balanced_accuracy": ""})
        row.update({f"recall_{class_name}": "" for class_name in class_names})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "experiment_comparison.csv", index=False)
    (output_dir / "experiment_plan.md").write_text(
        """# Eksperimen lanjutan yang disiapkan

Setiap eksperimen harus dijalankan terpisah dengan split train/validation dan checkpoint sendiri. Test set hanya dipakai sekali pada evaluasi akhir masing-masing eksperimen; jangan memilih konfigurasi berdasarkan test set. Kolom `not_run` sengaja dibiarkan kosong agar tidak ada angka fiktif. Metrik utama: macro F1-score dan balanced accuracy; recall per kelas tetap dilaporkan.

Urutan yang disarankan: class weight, focal loss, oversampling, augmentasi, CLAHE/kontras, lalu ukuran citra. Ubah satu faktor pada satu waktu dan simpan config, history, checkpoint, serta hasil test ke folder eksperimen yang berbeda.
""",
        encoding="utf-8",
    )


def write_readme(
    output_dir: Path,
    data_dir: Path,
    checkpoint: Path,
    class_names: list[str],
    split_counts: pd.Series,
    per_class: pd.DataFrame,
    overall: dict[str, Any],
    pairs: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    hardest = per_class.sort_values(["recall_sensitivity", "f1_score", "specificity"]).iloc[0]
    hardest_recall = per_class.sort_values("recall_sensitivity").iloc[0]
    hardest_f1 = per_class.sort_values("f1_score").iloc[0]
    top_pair = pairs.iloc[0] if not pairs.empty else None
    split_text = ", ".join(f"{label}: {int(count)}" for label, count in split_counts.items())
    leakage_note = leakage.loc[leakage["check"] == "patient_or_group_columns", "value"].iloc[0]
    table_columns = [
        "class",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "support",
        "roc_auc_ovr",
        "pr_auc_ovr",
    ]
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join("---" for _ in table_columns) + " |",
    ]
    for _, metric_row in per_class[table_columns].iterrows():
        cells = []
        for column in table_columns:
            value = metric_row[column]
            if column == "class":
                cells.append(str(value))
            elif column == "support":
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        table_lines.append("| " + " | ".join(cells) + " |")
    per_class_markdown = "\n".join(table_lines)
    text = f"# Evaluasi per kelas tumor otak\n\n"
    text += "Evaluasi ini hanya melakukan inference menggunakan checkpoint yang sudah ada; tidak ada training atau perubahan arsitektur.\n\n"
    text += "## Cara menjalankan\n\n"
    text += "```powershell\npython evaluate_per_class.py\npython evaluate_per_class.py --checkpoint hasil_edgenext\\20261029_182646\\02_tuning\\trial_1\\best_model.pt\n```\n\n"
    text += f"Output ditulis ke `{output_dir}`. Checkpoint yang dievaluasi: `{checkpoint}`.\n\n"
    text += "## Dataset, kelas, dan preprocessing\n\n"
    text += f"- Dataset: `{data_dir}`\n- Split yang dievaluasi: folder `test` resmi, {int(split_counts.sum())} gambar ({split_text})\n"
    text += f"- Kelas dari checkpoint/dataset: {', '.join(class_names)}\n"
    text += "- Preprocessing inference sama dengan notebook: crop foreground, pad menjadi persegi, resize 224, grayscale percentile 1–99, convert RGB, dan ImageNet normalization.\n"
    text += "- Model: EdgeNeXt dari `timm`; bobot dimuat dari `model_state_dict`, tanpa pretrained download ulang.\n\n"
    text += "## Metrik\n\n"
    text += "Per kelas dihitung precision, recall/sensitivity, specificity one-vs-rest dari TP/TN/FP/FN, F1-score, support, ROC-AUC OVR, dan PR-AUC. Ringkasan juga mencakup macro/weighted average, balanced accuracy, dan multiclass Matthews Correlation Coefficient.\n\n"
    text += "## Hasil setiap kelas\n\n"
    text += per_class_markdown + "\n\n"
    text += "## Ringkasan keseluruhan\n\n"
    text += f"- Accuracy: {overall['accuracy']:.4f}\n- Macro F1-score: {overall['macro_f1']:.4f}\n- Balanced accuracy: {overall['balanced_accuracy']:.4f}\n- MCC: {overall['matthews_correlation_coefficient']:.4f}\n- ROC-AUC OVR macro: {overall['roc_auc_ovr_macro']:.4f}\n- PR-AUC OVR macro: {overall['pr_auc_ovr_macro']:.4f}\n- Weighted F1-score: {overall['weighted_f1']:.4f}\n\n"
    text += f"## Kelas paling sulit\n\nBerdasarkan gabungan recall, F1, dan specificity terendah, kelas yang paling perlu diperhatikan adalah **{hardest['class']}**. Recall terendah: **{hardest_recall['class']}** ({hardest_recall['recall_sensitivity']:.4f}); F1 terendah: **{hardest_f1['class']}** ({hardest_f1['f1_score']:.4f}).\n\n"
    if top_pair is None:
        text += "Tidak ada kesalahan klasifikasi pada test set; tidak ada pasangan kelas tertukar.\n\n"
    else:
        text += f"Pasangan tertukar terbanyak adalah **{top_pair['true_label']} → {top_pair['predicted_label']}** sebanyak **{int(top_pair['count'])}** gambar. Rincian ada di `confusion_pairs.csv` dan folder `misclassified_by_pair/`.\n\n"
    text += "## Analisis kesalahan dan Grad-CAM\n\n"
    text += "`misclassified_examples.png` menampilkan contoh salah; `low_confidence_examples.png` menampilkan 20 confidence terendah; `misclassified_by_pair/` mengelompokkan kesalahan berdasarkan pasangan kelas; dan `gradcam/` berisi contoh benar, salah, serta confidence rendah jika layer Conv2d model mendukung Grad-CAM.\n\n"
    text += "## Integritas dan keterbatasan\n\n"
    raw_overlap = leakage.loc[leakage["check"] == "exact_sha256_overlap_train_test", "value"].iloc[0]
    effective_overlap = leakage.loc[leakage["check"] == "exact_sha256_overlap_effective_training_test", "value"].iloc[0]
    removed_rows = leakage.loc[leakage["check"] == "training_rows_removed_by_test_hash_deduplication", "value"].iloc[0]
    internal_removed = leakage.loc[leakage["check"] == "training_rows_removed_by_internal_hash_deduplication", "value"].iloc[0]
    effective_training_rows = leakage.loc[leakage["check"] == "training_rows_after_notebook_hash_cleanup", "value"].iloc[0]
    text += f"Audit SHA-256 train–test dicatat di `data_leakage_audit.csv`; overlap exact pada folder mentah: {raw_overlap}, overlap pada training efektif setelah deduplikasi test-hash: {effective_overlap}, baris train dikeluarkan karena hash test: {removed_rows}, duplikat internal train tambahan: {internal_removed}, sehingga referensi input training efektif menjadi {effective_training_rows} baris. Patient/group columns yang terdeteksi: `{leakage_note}`. Dataset ini tidak menyediakan identitas pasien yang jelas pada manifest, sehingga pemisahan berbasis pasien tidak dapat dibuktikan. Audit image-level tidak menjamin generalisasi ke pasien, scanner, atau rumah sakit baru.\n\n"
    text += "## Rekomendasi penelitian selanjutnya\n\n"
    text += "Gunakan `experiment_comparison.csv` dan jalankan satu perubahan per eksperimen: class weight, focal loss, augmentation, oversampling minoritas, CLAHE/contrast, kemudian ukuran citra. Pilih model berdasarkan validation macro F1, lalu laporkan test macro F1, balanced accuracy, recall tiap kelas, dan interval ketidakpastian bila memungkinkan. Tambahkan patient ID dan lakukan patient-level split pada rilis data berikutnya.\n"
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path best_model.pt; default: newest baseline checkpoint")
    parser.add_argument("--data-dir", type=Path, default=None, help="classification_task directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; default: results/per_class_evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta", action="store_true", help="Rata-rata prediksi citra asli + flip horizontal")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.90,
        help="Tandai prediksi dengan confidence < nilai ini sebagai needs_review. "
        "Gunakan ambang dari VALIDATION (metrics_test.json notebook: review_threshold), bukan dari test.",
    )
    parser.add_argument(
        "--gradcam-correct-samples",
        type=int,
        default=150,
        help="Jumlah sampel prediksi benar untuk pembanding metrik Grad-CAM-dalam-mask (0 = lewati)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    checkpoint = (args.checkpoint or find_default_checkpoint(project_root)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    data_dir = (args.data_dir or resolve_data_dir(project_root)).resolve()
    output_dir = (args.output_dir or project_root / "results" / "per_class_evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError("--batch-size harus positif")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda dipilih tetapi CUDA tidak tersedia")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    print(f"Project root : {project_root}")
    print(f"Dataset      : {data_dir}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Output       : {output_dir}")
    print(f"Device       : {device}")

    model, config, class_names, model_info = load_checkpoint(checkpoint, device)
    train_df = scan_split(data_dir, "train", class_names)
    test_df = scan_split(data_dir, "test", class_names)
    leakage = audit_data_leakage(project_root, data_dir, train_df, test_df)
    leakage.to_csv(output_dir / "data_leakage_audit.csv", index=False)
    raw_overlap = int(leakage.loc[leakage["check"] == "exact_sha256_overlap_train_test", "value"].iloc[0])
    effective_overlap = int(
        leakage.loc[leakage["check"] == "exact_sha256_overlap_effective_training_test", "value"].iloc[0]
    )
    if raw_overlap > 0:
        print(
            f"INFO: {raw_overlap} hash train-test ada pada folder mentah; "
            f"setelah aturan deduplikasi notebook, overlap training efektif={effective_overlap}."
        )
    if effective_overlap > 0:
        print("WARNING: overlap tetap ada pada training efektif; hasil perlu ditafsirkan hati-hati.")

    image_size = int(config.get("image_size", 224))
    transform = make_eval_transform(image_size)
    dataset = BrainTumorTestDataset(test_df, transform, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    y_true, y_pred, y_prob, paths, inference_seconds = predict(model, loader, device, use_tta=args.tta)
    per_class, report, overall, cm = calculate_metrics(y_true, y_pred, y_prob, class_names)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    report.to_csv(output_dir / "classification_report.csv", index=False)
    save_confusion_matrices(cm, class_names, output_dir)
    save_metric_plot(per_class, output_dir)
    save_curve_plots(y_true, y_prob, class_names, output_dir)
    predictions = make_prediction_frame(paths, y_true, y_pred, y_prob, class_names)
    predictions["needs_review"] = predictions["confidence"] < args.review_threshold
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    wrong, pairs, low_confidence = save_error_analysis(predictions, output_dir)
    flagged = predictions["needs_review"]
    review_summary = {
        "tta": bool(args.tta),
        "review_threshold": float(args.review_threshold),
        "review_flagged_count": int(flagged.sum()),
        "review_flagged_rate": float(flagged.mean()),
        "errors_caught_by_review": int((flagged & ~predictions["correct"]).sum()),
        "accuracy_not_flagged": float(predictions.loc[~flagged, "correct"].mean()) if (~flagged).any() else None,
    }

    overall.update(
        {
            "inference_seconds": inference_seconds,
            "inference_ms_per_image": inference_seconds / len(y_true) * 1000,
            "checkpoint": checkpoint,
            "data_dir": data_dir,
            "image_size": image_size,
            "class_names": class_names,
            "test_count": len(test_df),
            "misclassified_count": len(wrong),
            "lowest_confidence": float(low_confidence.iloc[0]["confidence"]),
            **review_summary,
        }
    )
    save_json(overall, output_dir / "summary_metrics.json")
    save_json(model_info, output_dir / "model_info.json")
    test_mask_dir = data_dir.parent / "segmentation_task" / "test" / "masks"
    save_gradcam_examples(
        model, predictions, transform, class_names, device, output_dir / "gradcam",
        mask_dir=test_mask_dir if test_mask_dir.is_dir() else None,
    )
    if args.gradcam_correct_samples > 0:
        save_gradcam_mask_overlap(
            model,
            predictions,
            transform,
            device,
            test_mask_dir,
            image_size,
            output_dir / "gradcam",
            correct_samples=args.gradcam_correct_samples,
        )
    save_experiment_plan(output_dir, class_names, overall, per_class)
    write_readme(
        output_dir,
        data_dir,
        checkpoint,
        class_names,
        test_df["true_label"].value_counts().sort_index(),
        per_class,
        overall,
        pairs,
        leakage,
    )

    print("\nPer-class metrics:")
    print(per_class.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nOverall:")
    for key in ["accuracy", "macro_f1", "balanced_accuracy", "matthews_correlation_coefficient", "roc_auc_ovr_macro", "pr_auc_ovr_macro"]:
        print(f"  {key}: {overall[key]:.6f}")
    print(f"\nMisclassified: {len(wrong)} / {len(predictions)}")
    print(
        f"Review (confidence < {args.review_threshold:.3f}): {review_summary['review_flagged_count']} ditandai, "
        f"menangkap {review_summary['errors_caught_by_review']} dari {len(wrong)} kesalahan"
    )
    if not pairs.empty:
        print("Most frequent confusion:", pairs.iloc[0].to_dict())
    print("Results written to:", output_dir)


if __name__ == "__main__":
    main()
