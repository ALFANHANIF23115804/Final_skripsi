"""Audit fungsi teknis dan kelengkapan eksperimen pipeline EdgeNeXt."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "edgenext_brisc2025_classification.ipynb"
RUNS_ROOT = PROJECT_ROOT / "hasil_edgenext"


def latest_run() -> Path:
    runs = [
        path
        for path in RUNS_ROOT.iterdir()
        if path.is_dir()
        and re.fullmatch(r"\d{8}_\d{6}", path.name)
        and (path / "01_baseline" / "best_model.pt").is_file()
        and (path / "02_tuning" / "tuning_results.csv").is_file()
        and (path / "03_final" / "predictions_test.csv").is_file()
    ]
    if not runs:
        raise FileNotFoundError(f"Tidak ada run di {RUNS_ROOT}")
    return max(runs, key=lambda path: path.name)


def check(label: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": label, "status": "PASS" if passed else "WARN", "detail": detail}


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8-sig"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    source = "\n".join(code_cells)
    for index, code in enumerate(code_cells, start=1):
        ast.parse(code, f"notebook_code_cell_{index}")

    run = latest_run()
    baseline = run / "01_baseline"
    tuning = run / "02_tuning"
    final = run / "03_final"
    comparison = run / "04_comparison"
    checks: list[dict[str, object]] = []

    required_outputs = [
        baseline / "best_model.pt",
        baseline / "metrics_test.json",
        baseline / "classification_report_test.csv",
        baseline / "predictions_test.csv",
        tuning / "tuning_results.csv",
        final / "metrics_test.json",
        final / "classification_report_test.csv",
        final / "predictions_test.csv",
        comparison / "baseline_vs_tuning.csv",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    checks.append(check("Output utama", not missing, "Semua output utama tersedia" if not missing else f"Hilang: {missing}"))

    tuning_table = pd.read_csv(tuning / "tuning_results.csv")
    checks.append(check("Tuning berjalan", len(tuning_table) > 0, f"{len(tuning_table)} trial tersimpan"))
    checks.append(
        check(
            "Pemilihan validation macro F1",
            "best_validation_macro_f1" in tuning_table.columns and "best_validation_macro_f1" in source,
            "Model dipilih berdasarkan validation macro F1",
        )
    )

    requested_strategies = {
        "class_weight": ["class_weight", "CrossEntropyLoss(weight"],
        "focal_loss": ["focal", "FocalLoss"],
        "oversampling": ["WeightedRandomSampler", "oversampling", "oversample"],
    }
    for strategy, tokens in requested_strategies.items():
        found = any(token in source for token in tokens)
        checks.append(check(f"Strategi {strategy}", found, "Terlihat di notebook" if found else "Belum diimplementasikan sebagai eksperimen"))

    config_paths = [baseline / "config.json", *sorted(tuning.glob("trial_*/config.json"))]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths if path.is_file()]
    for field in ["image_size", "drop_rate", "learning_rate", "weight_decay"]:
        values = sorted({str(config.get(field)) for config in configs})
        checks.append(
            check(
                f"Variasi eksperimen {field}",
                len(values) > 1,
                f"Nilai yang tersimpan: {values}" if len(values) > 1 else f"Hanya satu nilai: {values}; belum ada perbandingan faktor",
            )
        )

    seed_tokens = ["42", "43", "44"]
    seed_found = [seed for seed in seed_tokens if re.search(rf"\b{seed}\b", source)]
    checks.append(
        check(
            "Multi-seed 42/43/44",
            all(seed in seed_found for seed in seed_tokens),
            f"Seed yang terlihat di source: {seed_found}; belum berarti sudah dijalankan",
        )
    )

    per_config = []
    for path in [baseline / "config.json", *sorted(tuning.glob("trial_*/config.json")), final / "config.json"]:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            per_config.append((str(path), "random_seed" in data))
    missing_seed = [path for path, has_seed in per_config if not has_seed]
    checks.append(
        check(
            "Seed tersimpan per konfigurasi",
            not missing_seed,
            "Semua config menyimpan random_seed" if not missing_seed else f"Belum ada random_seed di: {missing_seed}",
        )
    )

    metrics = json.loads((final / "metrics_test.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(final / "predictions_test.csv")
    checks.append(
        check(
            "Prediksi dan metrik konsisten",
            len(predictions) == int(round(metrics.get("test_count", len(predictions))))
            if "test_count" in metrics
            else len(predictions) > 0,
            f"{len(predictions)} prediksi; accuracy={metrics.get('accuracy')}, macro_f1={metrics.get('macro_f1')}",
        )
    )

    report_lines = [
        "# Audit Pipeline Penelitian EdgeNeXt",
        "",
        f"Run terbaru: `{run}`",
        "",
        "## Status",
        "",
        "| Pemeriksaan | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        report_lines.append(f"| {item['check']} | {item['status']} | {item['detail']} |")
    report_lines += [
        "",
        "## Kesimpulan audit",
        "",
        "Pipeline teknis dapat membaca dataset, membuat DataLoader, membangun EdgeNeXt, melakukan training/evaluasi, dan menyimpan hasil. Namun status PASS teknis tidak berarti seluruh rancangan eksperimen lanjutan sudah selesai.",
        "",
        "Eksperimen class weight, focal loss, oversampling, augmentasi sebagai faktor terpisah, image-size comparison, dan validasi konsistensi multi-seed harus ditambahkan atau dijalankan sebelum menyimpulkan strategi optimasi yang konsisten.",
    ]
    output = comparison / "research_pipeline_audit.md"
    output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(f"[{item['status']}] {item['check']}: {item['detail']}" for item in checks))
    print(f"\nAudit tersimpan di: {output}")


if __name__ == "__main__":
    main()
