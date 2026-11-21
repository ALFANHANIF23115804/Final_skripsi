"""Memperbaiki cell manifest, DataLoader, dan Grad-CAM EdgeNeXt BRISC2025."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent


NOTEBOOK = Path(r"E:\CNN(Tumor otak)\Code Program\notebooks\edgenext_brisc2025_classification.ipynb")


def replace_source_fragment(cell, old, new, label):
    source = "".join(cell.get("source", []))
    if source.count(old) != 1:
        raise RuntimeError(f"Fragmen {label!r} harus muncul tepat satu kali.")
    cell["source"] = source.replace(old, new, 1).splitlines(keepends=True)


CELL_MANIFEST = dedent(
    '''
    # Cell 4 - Membaca manifest train dan test
    def detect_plane(filename):
        name = filename.lower()
        tokens = set(re.split(r"[_\\-.]+", name))
        if "axial" in name or "ax" in tokens:
            return "axial"
        if "coronal" in name or "co" in tokens:
            return "coronal"
        if "sagittal" in name or "sa" in tokens:
            return "sagittal"
        return "unknown"

    def file_sha256(image_path, chunk_size=1024 * 1024):
        digest = hashlib.sha256()
        with open(image_path, "rb") as file:
            for chunk in iter(lambda: file.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def scan_split(split_name):
        split_dir = DATA_DIR / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Folder split tidak ditemukan: {split_dir}")

        records = []
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_path = image_path.resolve()
                    records.append({
                        "image_path": str(image_path),
                        "split": split_name,
                        "label": class_name,
                        "label_id": CLASS_NAMES.index(class_name),
                        "filename": image_path.name,
                        "plane": detect_plane(image_path.name),
                        "file_sha256": file_sha256(image_path),
                    })

        if not records:
            raise RuntimeError(f"Tidak ada gambar yang ditemukan di {split_dir}")
        return pd.DataFrame(records)

    def validate_manifest_paths(dataframe, split_name, verify_images=False):
        """Validasi ulang path sebelum DataLoader dibuat atau training dimulai."""
        frame = dataframe.copy().reset_index(drop=True)
        missing = [str(value) for value in frame["image_path"] if not Path(value).is_file()]
        if missing:
            report_path = RUN_ROOT / f"missing_files_{split_name}.txt"
            report_path.write_text("\\n".join(missing), encoding="utf-8")
            examples = "\\n".join(missing[:5])
            raise FileNotFoundError(
                f"{split_name}: {len(missing)} file gambar tidak ditemukan sebelum training.\\n"
                f"Contoh:\\n{examples}\\n"
                f"Daftar file disimpan di: {report_path}"
            )

        if verify_images:
            corrupt = []
            for image_path in frame["image_path"]:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    corrupt.append({"image_path": image_path, "error": str(error)})
            if corrupt:
                report_path = RUN_ROOT / f"corrupt_files_{split_name}.csv"
                pd.DataFrame(corrupt).to_csv(report_path, index=False)
                raise RuntimeError(
                    f"{split_name}: {len(corrupt)} file gambar rusak. "
                    f"Daftar file disimpan di: {report_path}"
                )
        return frame

    train_all_df = scan_split("train")
    test_df = scan_split("test")
    train_all_df = validate_manifest_paths(train_all_df, "train_all", verify_images=True)
    test_df = validate_manifest_paths(test_df, "test", verify_images=True)

    # Pemeriksaan dan pencegahan data leakage berbasis hash gambar.
    # Folder test tidak diubah; salinan identik di train yang juga ada di test
    # dikeluarkan dari train agar evaluasi test tetap lebih jujur.
    test_hashes = set(test_df["file_sha256"])
    train_test_duplicate_df = train_all_df[train_all_df["file_sha256"].isin(test_hashes)].copy()
    train_without_test_duplicates = train_all_df[~train_all_df["file_sha256"].isin(test_hashes)].copy()
    train_duplicate_df = train_without_test_duplicates[
        train_without_test_duplicates.duplicated("file_sha256", keep="first")
    ].copy()
    train_all_df = train_without_test_duplicates.drop_duplicates("file_sha256", keep="first").reset_index(drop=True)

    quality_report = pd.DataFrame({
        "item": [
            "train_original",
            "train_test_exact_duplicates_removed",
            "train_internal_exact_duplicates_removed",
            "train_after_cleanup",
            "test_original_unchanged",
        ],
        "count": [
            len(train_all_df) + len(train_test_duplicate_df) + len(train_duplicate_df),
            len(train_test_duplicate_df),
            len(train_duplicate_df),
            len(train_all_df),
            len(test_df),
        ],
    })
    quality_report.to_csv(RUN_ROOT / "data_quality_report.csv", index=False)
    print("Duplikasi train-test yang dikeluarkan dari train:", len(train_test_duplicate_df))
    print("Duplikasi internal train yang dikeluarkan       :", len(train_duplicate_df))
    print("Jumlah train setelah pembersihan                :", len(train_all_df))

    print("Jumlah gambar train:", len(train_all_df))
    print("Jumlah gambar test :", len(test_df))
    display(pd.concat([
        train_all_df["label"].value_counts().rename("train"),
        test_df["label"].value_counts().rename("test")
    ], axis=1).fillna(0).astype(int))

    print("\\nDistribusi bidang:")
    display(pd.concat([
        train_all_df["plane"].value_counts().rename("train"),
        test_df["plane"].value_counts().rename("test")
    ], axis=1).fillna(0).astype(int))
    '''
).strip("\n")


CELL_DATALOADER = dedent(
    '''
    # Cell 8 - Dataset, transform, dan DataLoader
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def make_transforms(image_size, training=False):
        if training:
            return transforms.Compose([
                transforms.Lambda(lambda image: preprocess_brain_image(image, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomGamma(probability=0.35, gamma_range=(0.85, 1.15)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
                transforms.ToTensor(),
                AddGaussianNoise(probability=0.20, standard_deviation=0.015),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.Lambda(lambda image: preprocess_brain_image(image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    class BrainTumorDataset(Dataset):
        def __init__(self, dataframe, transform=None, image_size=IMAGE_SIZE):
            self.dataframe = dataframe.reset_index(drop=True).copy()
            self.transform = transform
            self.image_size = int(image_size)
            if self.image_size <= 0:
                raise ValueError(f"image_size harus positif, bukan {image_size}.")
            missing = [
                str(path) for path in self.dataframe["image_path"]
                if not Path(path).is_file()
            ]
            if missing:
                examples = "\\n".join(missing[:5])
                raise FileNotFoundError(
                    f"Dataset {len(missing)} file hilang saat DataLoader dibuat.\\n"
                    f"Contoh:\\n{examples}\\n"
                    "Jalankan ulang Cell 4 untuk membangun manifest terbaru."
                )

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]
            image_path = Path(row["image_path"])
            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
            except Exception as error:
                raise RuntimeError(f"Gagal membaca gambar: {image_path}. {error}") from error
            if self.transform is None:
                raise ValueError(
                    "Transform wajib diberikan agar gambar diubah menjadi tensor berukuran tetap."
                )
            image = self.transform(image)
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    f"Transform menghasilkan {type(image).__name__}, bukan torch.Tensor untuk {image_path}."
                )
            expected_shape = (3, self.image_size, self.image_size)
            if tuple(image.shape) != expected_shape:
                raise RuntimeError(
                    f"Preprocessing menghasilkan {tuple(image.shape)} untuk {image_path}; "
                    f"ukuran wajib {expected_shape}."
                )
            return image, int(row["label_id"]), str(image_path)

    def make_loaders(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, train_frame=None, val_frame=None, test_frame=None):
        image_size = int(image_size)
        batch_size = int(batch_size)
        if image_size <= 0 or batch_size <= 0:
            raise ValueError("image_size dan batch_size harus berupa bilangan positif.")

        frames = {
            "train": train_frame if train_frame is not None else globals().get("train_df"),
            "validation": val_frame if val_frame is not None else globals().get("val_df"),
            "test": test_frame if test_frame is not None else globals().get("test_df"),
        }
        missing_frames = [name for name, frame in frames.items() if frame is None]
        if missing_frames:
            raise RuntimeError(
                "DataFrame belum tersedia untuk: " + ", ".join(missing_frames) + ". "
                "Jalankan Cell 7 (Stratified split) terlebih dahulu atau kirim frame secara eksplisit."
            )

        train_frame = validate_manifest_paths(frames["train"], "train_loader")
        val_frame = validate_manifest_paths(frames["validation"], "val_loader")
        test_frame = validate_manifest_paths(frames["test"], "test_loader")
        train_dataset = BrainTumorDataset(train_frame, make_transforms(image_size, True), image_size=image_size)
        val_dataset = BrainTumorDataset(val_frame, make_transforms(image_size, False), image_size=image_size)
        test_dataset = BrainTumorDataset(test_frame, make_transforms(image_size, False), image_size=image_size)

        common = {
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        }
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
        return train_loader, val_loader, test_loader

    train_loader, val_loader, test_loader = make_loaders()
    images, labels, paths = next(iter(train_loader))
    print("Bentuk batch gambar:", images.shape)
    print("Bentuk label       :", labels.shape)
    print("Contoh path        :", paths[0])
    '''
).strip("\n")


CELL_GRADCAM = dedent(
    '''
    # Cell 17 - Grad-CAM dan perbandingan beberapa skala fitur
    def make_gradcam(model, image_path, transform, output_path, true_label=None, max_layers=4):
        was_training = model.training
        model.eval()
        output_path = Path(output_path)
        activations, gradients, handles = {}, {}, []

        def capture_activation(layer_name):
            def hook(_, __, output):
                if not isinstance(output, torch.Tensor) or output.ndim != 4:
                    return
                if min(output.shape[-2:]) < 2:
                    return
                activations[layer_name] = output

                def capture_gradient(gradient):
                    gradients[layer_name] = gradient.detach()

                if output.requires_grad:
                    output.register_hook(capture_gradient)
            return hook

        try:
            with Image.open(image_path) as source_image:
                original = source_image.convert("RGB")

            model_device = next(model.parameters()).device
            input_tensor = transform(original).unsqueeze(0).to(model_device)
            for layer_name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(capture_activation(layer_name)))

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(input_tensor)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise RuntimeError(f"Bentuk keluaran model tidak sesuai untuk satu gambar: {tuple(logits.shape)}")
                probabilities = torch.softmax(logits.detach(), dim=1)[0]
                predicted_class = int(probabilities.argmax().item())
                target_class = predicted_class
                target_confidence = float(probabilities[target_class].item())
                logits[0, target_class].backward()
            for handle in handles:
                handle.remove()
            handles.clear()

            if not activations:
                raise RuntimeError("Tidak ditemukan keluaran Conv2d spasial untuk Grad-CAM.")

            # Pilih layer terdalam pada setiap resolusi agar peta antar-tahap bisa dibandingkan.
            selected_layers = []
            seen_shapes = set()
            for layer_name in reversed(list(activations)):
                activation = activations[layer_name]
                spatial_shape = tuple(int(size) for size in activation.shape[-2:])
                if spatial_shape in seen_shapes:
                    continue
                if layer_name not in gradients:
                    continue
                selected_layers.append((layer_name, activation, gradients[layer_name]))
                seen_shapes.add(spatial_shape)
                if len(selected_layers) >= max_layers:
                    break
            selected_layers.reverse()

            cam_records = []
            for layer_name, activation, gradient in selected_layers:
                feature_map = activation[0].detach()
                gradient_map = gradient[0].detach()
                channel_weights = gradient_map.mean(dim=(1, 2), keepdim=True)
                raw_cam = torch.relu((channel_weights * feature_map).sum(dim=0))
                raw_peak = raw_cam.max()
                if not torch.isfinite(raw_peak) or raw_peak.item() <= 1e-12:
                    print(f"Layer dilewati karena CAM tidak memiliki aktivasi positif: {layer_name}")
                    continue

                normalized_cam = raw_cam / raw_peak
                resized_cam = torch.nn.functional.interpolate(
                    normalized_cam[None, None],
                    size=original.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].detach().cpu().numpy()
                flat_cam = resized_cam.ravel()
                top_count = max(1, int(np.ceil(flat_cam.size * 0.10)))
                total_mass = float(flat_cam.sum())
                top_mass_share = float(
                    np.partition(flat_cam, -top_count)[-top_count:].sum() / (total_mass + 1e-12)
                )
                cam_records.append({
                    "layer": layer_name,
                    "feature_height": int(feature_map.shape[-2]),
                    "feature_width": int(feature_map.shape[-1]),
                    "raw_cam_peak": float(raw_peak.item()),
                    "area_at_or_above_0_5": float(np.mean(resized_cam >= 0.5)),
                    "top_10_percent_activation_mass": top_mass_share,
                    "cam": resized_cam,
                })

            if not cam_records:
                raise RuntimeError("Semua layer menghasilkan CAM kosong/tidak positif; heatmap tidak dibuat.")

            # Bandingkan penghapusan area CAM teratas dengan beberapa area acak berukuran sama.
            rng = np.random.default_rng(42)
            input_height, input_width = input_tensor.shape[-2:]
            pixel_count = input_height * input_width
            removed_count = max(1, int(np.ceil(pixel_count * 0.10)))

            def target_probability_drop(pixel_indices):
                mask = torch.zeros(pixel_count, dtype=torch.bool, device=model_device)
                indices = torch.as_tensor(pixel_indices, dtype=torch.long, device=model_device)
                mask[indices] = True
                mask = mask.reshape(input_height, input_width)
                occluded_input = input_tensor.detach().clone()
                occluded_input[0, :, mask] = 0.0
                with torch.no_grad():
                    masked_probability = torch.softmax(model(occluded_input), dim=1)[0, target_class]
                return target_confidence - float(masked_probability.item())

            for record in cam_records:
                cam_tensor = torch.as_tensor(record["cam"], dtype=torch.float32, device=model_device)
                cam_tensor = torch.nn.functional.interpolate(
                    cam_tensor[None, None],
                    size=(input_height, input_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                top_indices = torch.topk(cam_tensor.reshape(-1), k=removed_count).indices.cpu().numpy()
                random_drops = [
                    target_probability_drop(rng.choice(pixel_count, size=removed_count, replace=False))
                    for _ in range(5)
                ]
                record["top10_probability_drop"] = target_probability_drop(top_indices)
                record["random_probability_drop_mean"] = float(np.mean(random_drops))
                record["random_probability_drop_std"] = float(np.std(random_drops))
                record["top10_more_influential_than_random"] = (
                    record["top10_probability_drop"] > record["random_probability_drop_mean"]
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_path = output_path.with_name(f"{output_path.stem}_layer_stats.csv")
            pd.DataFrame([
                {key: value for key, value in record.items() if key != "cam"}
                for record in cam_records
            ]).to_csv(comparison_path, index=False)

            figure = plt.figure(figsize=(4.0 * (len(cam_records) + 1) + 0.5, 8.2))
            grid = figure.add_gridspec(
                2,
                len(cam_records) + 2,
                width_ratios=[1.0] + [1.0] * len(cam_records) + [0.06],
                wspace=0.16,
                hspace=0.20,
            )
            original_axis = figure.add_subplot(grid[:, 0])
            original_axis.imshow(original)
            original_axis.axis("off")
            label_text = true_label if true_label is not None else "tidak tersedia"
            original_axis.set_title(
                f"Gambar asli\\nLabel: {label_text}\\n"
                f"Prediksi: {CLASS_NAMES[predicted_class]} ({target_confidence:.1%})",
                fontsize=9,
            )

            last_heatmap = None
            for column, record in enumerate(cam_records, start=1):
                cam = record["cam"]
                heatmap_axis = figure.add_subplot(grid[0, column])
                last_heatmap = heatmap_axis.imshow(
                    cam, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="bilinear"
                )
                heatmap_axis.axis("off")
                heatmap_axis.set_title(
                    f"Heatmap\\n{record['layer']}\\n"
                    f"fitur {record['feature_height']}x{record['feature_width']}",
                    fontsize=8,
                )
                overlay_axis = figure.add_subplot(grid[1, column])
                overlay_axis.imshow(original)
                overlay_axis.imshow(
                    cam,
                    cmap="inferno",
                    vmin=0.0,
                    vmax=1.0,
                    alpha=np.clip(cam * 0.80, 0.0, 0.80),
                    interpolation="bilinear",
                )
                overlay_axis.axis("off")
                overlay_axis.set_title(
                    f"Overlay | area >= 0.5: {record['area_at_or_above_0_5']:.1%}\\n"
                    f"Δp top 10%: {record['top10_probability_drop']:+.1%} | "
                    f"random: {record['random_probability_drop_mean']:+.1%}",
                    fontsize=8,
                )

            colorbar_axis = figure.add_subplot(grid[0, -1])
            if last_heatmap is None:
                raise RuntimeError("Tidak ada heatmap yang dapat ditampilkan.")
            colorbar = figure.colorbar(
                last_heatmap,
                cax=colorbar_axis,
                label="Aktivasi relatif\\n(0-1)",
            )
            colorbar.ax.tick_params(labelsize=8)
            figure.suptitle(
                "Grad-CAM: perbandingan layer spasial untuk kelas "
                f"{CLASS_NAMES[predicted_class]}",
                fontsize=12,
            )
            figure.text(
                0.5,
                0.01,
                "Normalisasi hanya untuk visualisasi; Δp membandingkan occlusion terpilih dengan acak, bukan akurasi lokalisasi.",
                ha="center",
                fontsize=8,
            )
            figure.subplots_adjust(top=0.88, bottom=0.08, left=0.02, right=0.99)
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            if "agg" not in plt.get_backend().lower():
                plt.show()
            plt.close(figure)

            print("Prediksi:", CLASS_NAMES[predicted_class])
            print("Confidence prediksi:", f"{target_confidence:.1%}")
            print("Label sebenarnya:", label_text)
            print("Layer dibandingkan:", [record["layer"] for record in cam_records])
            print("Perubahan confidence top-10% vs acak:")
            for record in cam_records:
                print(
                    f"  {record['layer']}: {record['top10_probability_drop']:+.1%} vs "
                    f"{record['random_probability_drop_mean']:+.1%} "
                    f"(lebih berpengaruh: {record['top10_more_influential_than_random']})"
                )
            print("Statistik layer tersimpan di:", comparison_path)
            return {
                "prediction": CLASS_NAMES[predicted_class],
                "confidence": target_confidence,
                "true_label": true_label,
                "layers": pd.DataFrame([
                    {key: value for key, value in record.items() if key != "cam"}
                    for record in cam_records
                ]),
                "image_path": output_path,
                "stats_path": comparison_path,
            }
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    gradcam_predictions = final_evaluation["predictions"]
    if gradcam_predictions.empty:
        raise RuntimeError("Tidak ada prediksi test untuk contoh Grad-CAM.")
    sample_row = gradcam_predictions.iloc[0]
    gradcam_result = make_gradcam(
        final_result["model"],
        sample_row["image_path"],
        make_transforms(final_config["image_size"], False),
        COMPARISON_DIR / "gradcam_example.png",
        true_label=sample_row["true_label"],
    )
    '''
).strip("\n")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        snapshot_path = Path(sys.argv[1])
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if len(snapshot["cells"]) != len(notebook["cells"]):
            raise RuntimeError("Salinan sesi memiliki jumlah cell yang berbeda; merge dibatalkan.")
        for index, (cell, saved_cell) in enumerate(zip(notebook["cells"], snapshot["cells"])):
            if "".join(cell.get("source", [])) != "".join(saved_cell.get("source", [])):
                raise RuntimeError(
                    f"Source cell {index + 1} berbeda; merge output dibatalkan agar kode tidak tertimpa."
                )
        for cell, saved_cell in zip(notebook["cells"], snapshot["cells"]):
            cell["outputs"] = saved_cell.get("outputs", [])
            cell["execution_count"] = saved_cell.get("execution_count")

    replaced = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "# Cell 1 - Import library dan pemeriksaan environment" in source:
            if 'print("CUDA build   :", torch.version.cuda)' in source:
                replace_source_fragment(
                    cell,
                    'print("CUDA build   :", torch.version.cuda)',
                    'print("CUDA build   :", getattr(getattr(torch, "version", None), "cuda", None))',
                    "CUDA version lookup",
                )
            replaced.append("environment")
        elif "# Cell 4 - Membaca manifest train dan test" in source:
            cell["source"] = CELL_MANIFEST.splitlines(keepends=True)
            replaced.append("manifest")
        elif "# Cell 8 - Dataset, transform, dan DataLoader" in source:
            cell["source"] = CELL_DATALOADER.splitlines(keepends=True)
            replaced.append("dataloader")
        elif "# Cell 17 - Grad-CAM" in source:
            cell["source"] = CELL_GRADCAM.splitlines(keepends=True)
            replaced.append("gradcam")
    if sorted(replaced) != ["dataloader", "environment", "gradcam", "manifest"]:
        raise RuntimeError(f"Cell yang diperbaiki tidak lengkap: {replaced}")

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook diperbaiki: {NOTEBOOK}")
    print(f"Cell diperbaiki: {', '.join(replaced)}")


if __name__ == "__main__":
    main()
