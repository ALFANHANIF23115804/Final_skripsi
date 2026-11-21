"""Analisis kesalahan pasangan kelas dari hasil evaluasi EdgeNeXt yang sudah ada."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def detect_plane(path: str) -> str:
    tokens = set(re.split(r"[_\-.]+", Path(path).name.lower()))
    if "axial" in tokens or "ax" in tokens:
        return "axial"
    if "coronal" in tokens or "co" in tokens:
        return "coronal"
    if "sagittal" in tokens or "sa" in tokens:
        return "sagittal"
    return "unknown"


def latest_run(root: Path) -> Path:
    runs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"\d{8}_\d{6}", path.name)
        and (path / "03_final" / "predictions_test.csv").is_file()
    ]
    if not runs:
        raise FileNotFoundError(f"Tidak ada run ditemukan di {root}")
    return max(runs, key=lambda path: path.name)


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="Folder run hasil_edgenext; default: run terbaru")
    parser.add_argument("--output", type=Path, default=None, help="Folder output analisis")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "hasil_edgenext"
    run = (args.run or latest_run(runs_root)).resolve()
    final_dir = run / "03_final"
    comparison_dir = run / "04_comparison"
    output = (args.output or comparison_dir / "error_pair_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(final_dir / "predictions_test.csv")
    predictions["plane"] = predictions["image_path"].map(detect_plane)
    predictions["confidence_band"] = pd.cut(
        predictions["confidence"],
        bins=[-float("inf"), 0.70, 0.90, float("inf")],
        labels=["low_<0.70", "medium_0.70-0.90", "high_>0.90"],
    )
    wrong = predictions[~predictions["correct"]].copy()
    wrong.to_csv(output / "misclassified_detailed.csv", index=False)

    pair = (
        wrong.groupby(["true_label", "predicted_label"], as_index=False)
        .agg(
            errors=("image_path", "size"),
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        )
        .sort_values("errors", ascending=False)
    )
    pair.to_csv(output / "error_pairs_detailed.csv", index=False)

    pair_plane = (
        wrong.groupby(["true_label", "predicted_label", "plane"], as_index=False)
        .size()
        .rename(columns={"size": "errors"})
        .sort_values("errors", ascending=False)
    )
    pair_plane.to_csv(output / "error_pairs_by_plane.csv", index=False)

    confidence = (
        wrong.groupby("confidence_band", observed=False, as_index=False)
        .size()
        .rename(columns={"size": "errors"})
    )
    confidence.to_csv(output / "error_confidence_bands.csv", index=False)

    metrics = load_metrics(final_dir / "metrics_test.json")
    baseline_metrics = load_metrics(run / "01_baseline" / "metrics_test.json")
    tuning_path = run / "02_tuning" / "tuning_results.csv"
    tuning = pd.read_csv(tuning_path) if tuning_path.is_file() else pd.DataFrame()

    report_lines = [
        "# Analisis Kesalahan Pasangan Kelas EdgeNeXt",
        "",
        f"Run: `{run}`",
        "",
        "## Ringkasan hasil nyata",
        "",
        f"- Baseline accuracy: `{baseline_metrics.get('accuracy', 'tidak tersedia')}`",
        f"- Final accuracy: `{metrics.get('accuracy', 'tidak tersedia')}`",
        f"- Baseline macro F1: `{baseline_metrics.get('macro_f1', 'tidak tersedia')}`",
        f"- Final macro F1: `{metrics.get('macro_f1', 'tidak tersedia')}`",
        f"- Total kesalahan final: `{len(wrong)}` dari `{len(predictions)}` gambar",
        "",
        "## Pasangan kelas yang tertukar",
        "",
        "| Kelas sebenarnya | Prediksi | Jumlah | Confidence rata-rata |",
        "|---|---|---:|---:|",
    ]
    if pair.empty:
        report_lines.append("| Tidak ada | Tidak ada | 0 | - |")
    else:
        for row in pair.itertuples(index=False):
            report_lines.append(
                f"| {row.true_label} | {row.predicted_label} | {int(row.errors)} | {row.mean_confidence:.4f} |"
            )
    report_lines += [
        "",
        "## Interpretasi",
        "",
        "- Pasangan dengan jumlah error terbesar menjadi prioritas analisis visual dan eksperimen optimasi.",
        "- Confidence tinggi pada prediksi salah menunjukkan kesalahan yang lebih sistematis; confidence rendah menunjukkan contoh ambigu atau sulit.",
        "- Distribusi orientasi hanya bersifat indikatif karena orientasi diperoleh dari token nama file.",
        "- Grad-CAM dapat digunakan untuk melihat area yang memengaruhi model, tetapi bukan bukti segmentasi tumor.",
        "",
        "## Status konsistensi optimasi",
        "",
        "Hasil ini belum membuktikan konsistensi lintas random seed. Konsistensi harus diuji dengan menjalankan konfigurasi terbaik pada beberapa seed dan melaporkan rata-rata serta standar deviasi macro F1, recall kelas sulit, dan jumlah error pasangan.",
        "",
        "## Artefak",
        "",
        "- `misclassified_detailed.csv`: seluruh prediksi salah dengan confidence dan orientasi.",
        "- `error_pairs_detailed.csv`: ringkasan error per pasangan kelas.",
        "- `error_pairs_by_plane.csv`: error per pasangan dan orientasi.",
        "- `error_confidence_bands.csv`: error berdasarkan rentang confidence.",
    ]
    (output / "error_pair_analysis.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Run dianalisis: {run}")
    print(f"Total error: {len(wrong)} / {len(predictions)}")
    print(pair.to_string(index=False) if not pair.empty else "Tidak ada error")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
