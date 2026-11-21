"""Audit and prepare the Mendeley four-class external dataset.

The script is deliberately conservative: modality and patient metadata are not
invented. Images without explicit T1 evidence remain ``unknown`` and are not
eligible for a strict external evaluation until the manifest is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupKFold


CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_ALIASES = {
    "glioma": "glioma",
    "healthy": "no_tumor",
    "healthy_brain": "no_tumor",
    "no_tumor": "no_tumor",
    "normal": "no_tumor",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary_macroadenoma": "pituitary",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUGMENTATION_TOKENS = (
    "aug",
    "flip",
    "rotate",
    "rotated",
    "noise",
    "blur",
    "synthetic",
    "generated",
    "mixup",
    "cutmix",
)
MODALITY_TOKENS = {
    "t1": "t1_candidate",
    "t1c": "t1_candidate",
    "t1ce": "t1_candidate",
    "contrast": "t1_candidate",
    "gad": "t1_candidate",
    "t2": "not_t1",
    "flair": "not_t1",
    "dwi": "not_t1",
    "adc": "not_t1",
}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_2d(values: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    positions = np.arange(size, dtype=np.float32)
    frequencies = np.arange(size, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / size * (positions + 0.5) * frequencies)
    return basis @ values @ basis.T


def phash_hex(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Return a perceptual hash based on low-frequency DCT coefficients."""

    size = hash_size * highfreq_factor
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    coefficients = _dct_2d(values)[:hash_size, :hash_size]
    low_frequency = coefficients[1:, 1:]
    median = float(np.median(low_frequency))
    bits = (low_frequency > median).reshape(-1)
    bit_string = "".join("1" if bit else "0" for bit in bits)
    return f"{int(bit_string, 2):0{(hash_size * hash_size - 1 + 3) // 4}x}"


def phash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_class_root(root: Path) -> Path:
    """Find the directory whose children are the dataset class folders."""

    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        names = {normalize_name(path.name) for path in candidate.iterdir() if path.is_dir()}
        if len(names & set(CLASS_ALIASES)) >= 3:
            return candidate
    raise FileNotFoundError(f"Folder kelas tidak ditemukan di {root}")


def modality_status(path: Path) -> str:
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    for token in tokens:
        if token in MODALITY_TOKENS:
            return MODALITY_TOKENS[token]
    return "unknown"


def augmentation_status(path: Path) -> str:
    text = path.as_posix().lower()
    return "suspected_augmentation" if any(token in text for token in AUGMENTATION_TOKENS) else "not_detected"


def scan_external(root: Path) -> pd.DataFrame:
    class_root = find_class_root(root)
    records: list[dict] = []
    for class_dir in sorted(class_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = CLASS_ALIASES.get(normalize_name(class_dir.name))
        if label is None:
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                records.append(
                    {
                        "dataset": "mendeley_82mtzd8x72_v1",
                        "image_path": str(path),
                        "filename": path.name,
                        "label": label,
                        "modality_status": modality_status(path),
                        "augmentation_status": augmentation_status(path),
                        "group_id": "unknown",
                        "group_status": "missing_patient_id",
                        "sha256": file_sha256(path),
                        "phash": phash_hex(path),
                    }
                )
            except Exception as error:
                print(f"File dilewati karena rusak: {path} ({error})")
    return pd.DataFrame(records)


def scan_brisc(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    is_mask = manifest["is_mask"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    manifest = manifest[(manifest["task"] == "classification") & (~is_mask)].copy()
    root = manifest_path.parent
    records: list[dict] = []
    for row in manifest.itertuples(index=False):
        path = root / str(row.relative_path).replace("\\", "/")
        if not path.exists():
            continue
        records.append(
            {
                "dataset": "BRISC2025",
                "image_path": str(path),
                "filename": path.name,
                "label": str(row.tumor_label),
                "sha256": str(row.sha256),
                "phash": phash_hex(path),
            }
        )
    return pd.DataFrame(records)


def audit_phash(external: pd.DataFrame, brisc: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """Match each external image to its closest BRISC image by pHash."""

    result = external.copy()
    if external.empty or brisc.empty:
        result["nearest_phash_distance"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["exact_sha256_overlap"] = False
        result["phash_overlap"] = False
        result["eligible_after_overlap"] = True
        result["eligible_for_strict_eval"] = False
        return result
    internal_hashes = brisc["phash"].tolist()
    internal_sha = set(brisc["sha256"])
    distances: list[int] = []
    exact: list[bool] = []
    near: list[bool] = []
    for row in external.itertuples(index=False):
        best = min(phash_distance(row.phash, candidate) for candidate in internal_hashes)
        distances.append(best)
        exact.append(row.sha256 in internal_sha)
        near.append(best <= threshold)
    result["nearest_phash_distance"] = distances
    result["exact_sha256_overlap"] = exact
    result["phash_overlap"] = near
    result["eligible_after_overlap"] = ~(result["exact_sha256_overlap"] | result["phash_overlap"])
    result["eligible_for_strict_eval"] = (
        result["eligible_after_overlap"]
        & result["modality_status"].eq("t1_verified")
        & result["augmentation_status"].eq("not_detected")
        & result["group_status"].eq("verified_patient_id")
    )
    return result


def apply_verified_metadata(frame: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    """Apply a human-verified modality/group table without guessing missing values."""

    result = frame.copy()
    if metadata_path is None:
        return result
    metadata = pd.read_csv(metadata_path)
    required = {"image_path", "modality_status", "group_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "Metadata verifikasi wajib memiliki kolom: " + ", ".join(sorted(required))
            + ". Kurang: " + ", ".join(sorted(missing))
        )
    metadata = metadata.drop_duplicates("image_path", keep="last").copy()
    metadata["image_path"] = metadata["image_path"].map(lambda value: str(Path(value).resolve()))
    result["image_path"] = result["image_path"].map(lambda value: str(Path(value).resolve()))
    result = result.drop(columns=["modality_status", "group_id", "group_status"], errors="ignore")
    result = result.merge(
        metadata[["image_path", "modality_status", "group_id"] + (["augmentation_status"] if "augmentation_status" in metadata else [])],
        on="image_path",
        how="left",
        suffixes=("", "_verified"),
    )
    result["modality_status"] = result["modality_status"].replace("", np.nan).fillna("unknown")
    result["group_id"] = result["group_id"].replace("", np.nan).fillna("unknown")
    result["group_status"] = np.where(result["group_id"].eq("unknown"), "missing_patient_id", "verified_patient_id")
    if "augmentation_status_verified" in result:
        result["augmentation_status"] = result["augmentation_status_verified"].fillna(result["augmentation_status"])
        result = result.drop(columns=["augmentation_status_verified"])
    return result


def mark_strict_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep strict external evaluation closed unless every required audit passes."""

    result = frame.copy()
    required_overlap = result.get("eligible_after_overlap", pd.Series(False, index=result.index))
    result["eligible_for_strict_eval"] = (
        required_overlap
        & result["modality_status"].eq("t1_verified")
        & result["augmentation_status"].eq("not_detected")
        & result["group_status"].eq("verified_patient_id")
    )
    def reasons(row: pd.Series) -> str:
        values = []
        if row.get("augmentation_status") == "suspected_augmentation":
            values.append("suspected_augmentation")
        if row.get("modality_status") != "t1_verified":
            values.append("t1_contrast_not_verified")
        if not bool(row.get("eligible_after_overlap", False)):
            values.append("overlap_with_brisc")
        if row.get("group_status") != "verified_patient_id":
            values.append("patient_group_missing")
        return ";".join(values)

    result["keep_for_evaluation"] = result["eligible_for_strict_eval"]
    result["exclusion_reason"] = result.apply(reasons, axis=1)
    return result


def crop_brain_pad_square(image: Image.Image) -> Image.Image:
    """Remove mostly-black border, pad the brain crop to square, and return RGB."""

    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    low, high = np.percentile(gray, [5, 99])
    threshold = max(3.0, low + 0.12 * (high - low))
    foreground = gray > threshold
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if len(rows) and len(cols):
        image = image.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    side = max(image.size)
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    return ImageOps.expand(image, border=(left, top, side - image.width - left, side - image.height - top), fill=0).convert("RGB")


def normalize_intensity(image: Image.Image, size: int = 224) -> Image.Image:
    """Apply robust per-image percentile normalization after brain cropping."""

    image = crop_brain_pad_square(image).resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def make_group_kfold(frame: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create leakage-resistant folds; refuse to fake patient groups."""

    if "group_id" not in frame or frame["group_id"].eq("unknown").any():
        raise ValueError("GroupKFold diblokir: patient_id/group_id belum tersedia untuk semua gambar.")
    if frame["group_id"].nunique() < n_splits:
        raise ValueError("Jumlah group unik lebih kecil daripada jumlah fold.")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(frame, frame["label"], groups=frame["group_id"]))


def write_reports(external: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    external.to_csv(output_dir / "mendeley_manifest.csv", index=False)
    template = external[["image_path", "label"]].copy()
    template["modality_status"] = "unknown"
    template["group_id"] = "unknown"
    template["augmentation_status"] = external["augmentation_status"]
    template["keep_for_evaluation"] = external.get("keep_for_evaluation", False)
    template.to_csv(output_dir / "verified_metadata_template.csv", index=False)
    grouped = external.groupby(["dataset", "label"], dropna=False)
    report = grouped.size().rename("raw_count").reset_index()
    report["valid_image_count"] = report["raw_count"]
    report["after_augmentation_filter"] = (
        grouped["augmentation_status"].apply(lambda values: (values != "suspected_augmentation").sum()).to_numpy()
    )
    report["after_overlap_filter"] = grouped["eligible_after_overlap"].sum().to_numpy()
    report["suspected_augmentation"] = (
        grouped["augmentation_status"].apply(lambda values: (values == "suspected_augmentation").sum()).to_numpy()
    )
    report["t1_verified"] = (
        grouped["modality_status"].apply(lambda values: (values == "t1_verified").sum()).to_numpy()
    )
    report["phash_overlap"] = grouped["phash_overlap"].sum().to_numpy()
    report["exact_sha256_overlap"] = grouped["exact_sha256_overlap"].sum().to_numpy()
    report["strict_eval_eligible"] = grouped["eligible_for_strict_eval"].sum().to_numpy()
    report["strict_eval_excluded"] = report["raw_count"] - report["strict_eval_eligible"]
    report.to_csv(output_dir / "report_per_dataset_per_class.csv", index=False)
    if external.empty:
        modality_counts = {}
        augmentation_count = 0
        strict_count = 0
    else:
        modality_counts = external["modality_status"].value_counts(dropna=False).to_dict()
        augmentation_count = int((external["augmentation_status"] == "suspected_augmentation").sum())
        strict_count = int(external.get("eligible_for_strict_eval", pd.Series(False, index=external.index)).sum())
    summary = {
        "source": "https://data.mendeley.com/datasets/82mtzd8x72/1",
        "doi": "10.17632/82mtzd8x72.1",
        "raw_images_scanned": int(len(external)),
        "classes": CLASS_NAMES,
        "modality_counts": {str(key): int(value) for key, value in modality_counts.items()},
        "t1_verified": int(modality_counts.get("t1_verified", 0)),
        "t1_candidates": int(modality_counts.get("t1_candidate", 0)),
        "modality_unknown": int(modality_counts.get("unknown", 0)),
        "augmentation_suspected": augmentation_count,
        "strict_evaluation_eligible": strict_count,
        "patient_id_available": False,
        "evaluation_status": "blocked_until_T1_and_patient_metadata_are_verified",
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--brisc-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument(
        "--verified-metadata",
        type=Path,
        default=None,
        help="CSV manual berisi image_path, modality_status=t1_verified, group_id, dan opsional augmentation_status.",
    )
    args = parser.parse_args()

    external = scan_external(args.external_root)
    external = apply_verified_metadata(external, args.verified_metadata)
    brisc = scan_brisc(args.brisc_manifest)
    audited = audit_phash(external, brisc, args.phash_threshold)
    audited = mark_strict_eligibility(audited)
    write_reports(audited, args.output_dir)
    print(f"Mendeley images terbaca: {len(audited)}")
    print(audited.groupby("label").size().to_string())
    print(f"pHash overlap <= {args.phash_threshold}: {int(audited['phash_overlap'].sum())}")
    print(f"SHA-256 overlap: {int(audited['exact_sha256_overlap'].sum())}")
    print(f"Eligible strict evaluation: {int(audited.get('eligible_for_strict_eval', pd.Series(dtype=bool)).sum())}")
    print("Evaluasi strict tetap diblokir sampai T1 kontras dan group pasien diverifikasi.")


if __name__ == "__main__":
    main()
