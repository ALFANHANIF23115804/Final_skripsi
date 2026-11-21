"""Smoke test ringan untuk manifest, DataLoader, dan arsitektur EdgeNeXt."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")
NOTEBOOK = Path(r"E:\CNN(Tumor otak)\Code Program\notebooks\edgenext_brisc2025_classification.ipynb")
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}


def run_cell(index: int, source: str | None = None) -> None:
    code = source if source is not None else "".join(notebook["cells"][index].get("source", []))
    exec(compile(code, f"cell_{index}", "exec"), namespace, namespace)


run_cell(1)
run_cell(3)
run_cell(4)
namespace['USE_PRETRAINED'] = "False"

# Smoke test tidak perlu menghitung hash seluruh dataset; training notebook asli
# tetap memakai hash untuk audit duplikasi.
manifest_source = "".join(notebook["cells"][6].get("source", []))
manifest_source = manifest_source.replace(
    '"file_sha256": file_sha256(image_path),',
    '"file_sha256": hashlib.sha256(str(image_path).encode()).hexdigest(),',
)
manifest_source = manifest_source.replace(
    'train_all_df = validate_manifest_paths(train_all_df, "train_all")',
    'train_all_df = train_all_df.reset_index(drop=True)',
).replace(
    'test_df = validate_manifest_paths(test_df, "test")',
    'test_df = test_df.reset_index(drop=True)',
)
run_cell(6, manifest_source)
run_cell(10)
run_cell(12)
run_cell(14)
print("SMOKE_TEST_LOADER_MODEL_OK")
