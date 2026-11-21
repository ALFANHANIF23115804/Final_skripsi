"""Membuat panduan PDF penelitian EdgeNeXt BRISC2025 dalam bahasa Indonesia."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(r"E:\CNN(Tumor otak)\Code Program")
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "edgenext_brisc2025_classification.ipynb"
DATASET_DIR = PROJECT_ROOT / "brisc2025-v6" / "brisc2025" / "classification_task"
PREVIOUS_GRADCAM = Path(r"C:\Users\alfan\AppData\Local\Temp\codex-clipboard-14b0ef1b-73d7-484a-9ff4-e214062142a9.png")
TMP_ROOT = PROJECT_ROOT / "tmp" / "pdfs" / "edgenext_guide"
SCREENSHOT_DIR = TMP_ROOT / "cell_screenshots"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "panduan_alur_penelitian_edgenext_cell_1-19.pdf"


CELL_INFO = {
    1: {
        "title": "Import library dan pemeriksaan environment",
        "topic": "Menyiapkan semua library serta mengecek versi Python, PyTorch, CUDA, dan GPU.",
        "function": "Memastikan notebook berjalan pada Python 3.10 dan mengetahui apakah training dapat memakai GPU.",
        "output": "Versi library, status CUDA, nama GPU, dan penggunaan memory awal.",
        "note": "Jika CUDA tersedia, DEVICE menjadi cuda. Jika tidak, model berjalan memakai CPU dan biasanya lebih lambat.",
    },
    2: {
        "title": "Konfigurasi penelitian",
        "topic": "Menentukan lokasi dataset, folder hasil, ukuran gambar, batch size, epoch, learning rate, dan nama model.",
        "function": "Menyediakan satu tempat untuk mengubah pengaturan penelitian tanpa mengubah banyak cell.",
        "output": "DATA_DIR, OUTPUT_ROOT, RUN_ROOT, CLASS_NAMES, DEVICE, dan parameter eksperimen.",
        "note": "Dataset BRISC2025 sudah JPG 2D, sehingga tidak ada konversi 3D ke 2D.",
    },
    3: {
        "title": "Fungsi bantuan dan seed",
        "topic": "Membuat fungsi untuk seed, pengukuran memory, RAM, dan penyimpanan JSON.",
        "function": "Membantu eksperimen lebih konsisten dan menyimpan konfigurasi tanpa error tipe data seperti torch.device.",
        "output": "Fungsi bantuan yang dipakai oleh cell training dan penyimpanan hasil.",
        "note": "Random seed membantu hasil lebih mudah diulang, walaupun hasil GPU dapat sedikit berbeda.",
    },
    4: {
        "title": "Membaca manifest train dan test",
        "topic": "Membaca semua gambar pada folder train dan test, memberi label kelas, mendeteksi bidang gambar, dan menghitung hash.",
        "function": "Membangun tabel data yang menjadi sumber utama untuk split, DataLoader, dan audit duplikasi.",
        "output": "train_all_df dan test_df yang berisi path gambar, label, label_id, plane, serta file_sha256.",
        "note": "Path divalidasi sebelum DataLoader. Jika ada file hilang atau rusak, dibuat laporan agar error tidak muncul di tengah training.",
    },
    5: {
        "title": "Distribusi kelas dan pemeriksaan file rusak",
        "topic": "Membuat grafik distribusi kelas dan memeriksa apakah gambar dapat dibuka.",
        "function": "Mengetahui apakah dataset seimbang dan memastikan tidak ada file yang rusak.",
        "output": "Grafik distribusi, jumlah file rusak, dan file corrupt_images.csv jika ada.",
        "note": "Distribusi kelas penting karena accuracy saja dapat menyesatkan jika jumlah kelas tidak seimbang.",
    },
    6: {
        "title": "Menampilkan contoh gambar",
        "topic": "Menampilkan beberapa gambar dari setiap kelas.",
        "function": "Pemeriksaan visual untuk memastikan gambar sesuai kelas dan tidak salah membaca folder.",
        "output": "Grid contoh gambar yang disimpan sebagai sample_images.png.",
        "note": "Pemeriksaan visual membantu menemukan gambar kosong, terlalu gelap, atau label folder yang keliru.",
    },
    7: {
        "title": "Pembagian data train dan validation",
        "topic": "Membagi data train menjadi 80 persen train dan 20 persen validation menggunakan stratified split.",
        "function": "Membuat data untuk belajar dan data untuk memantau kemampuan generalisasi selama training.",
        "output": "train_df, val_df, test_df, serta tabel split_summary.csv.",
        "note": "Folder test tidak diubah dan hanya dipakai pada evaluasi akhir.",
    },
    8: {
        "title": "Preprocessing, Dataset, dan DataLoader",
        "topic": "Mengubah gambar menjadi RGB, resize 224x224, normalisasi, augmentasi ringan, dan membentuk batch.",
        "function": "Menyiapkan input yang sesuai untuk EdgeNeXt dan mengirimkannya secara efisien ke GPU atau CPU.",
        "output": "train_loader, val_loader, test_loader, serta bentuk batch tensor.",
        "note": "Augmentasi hanya digunakan pada train. Validation dan test menggunakan transformasi tanpa augmentasi acak.",
    },
    9: {
        "title": "Membuat model EdgeNeXt",
        "topic": "Memanggil EdgeNeXt dari timm dan mengganti classifier terakhir menjadi empat kelas.",
        "function": "Membangun model CNN ringan yang belajar membedakan glioma, meningioma, pituitary, dan no_tumor.",
        "output": "Arsitektur model, jumlah parameter, parameter trainable, dan ukuran model.",
        "note": "Penelitian ini memakai satu keluarga model saja. Perbandingan dilakukan sebelum dan sesudah tuning.",
    },
    10: {
        "title": "Fungsi training",
        "topic": "Mendefinisikan run_epoch dan fit_model untuk training, validation, early stopping, scheduler, dan penyimpanan model.",
        "function": "Menjalankan proses belajar model dan memilih bobot terbaik berdasarkan validation macro F1-score.",
        "output": "History training, best_model.pt, history.csv, resource_metrics.json, dan grafik training.",
        "note": "Early stopping dapat menghentikan training sebelum epoch maksimum jika validation tidak membaik.",
    },
    11: {
        "title": "Fungsi evaluasi test",
        "topic": "Menghitung prediksi, confidence, accuracy, precision, recall, macro F1, ROC-AUC, report, dan confusion matrix.",
        "function": "Mengukur kemampuan model pada data yang tidak dipakai untuk memperbarui bobot.",
        "output": "metrics_test.json, classification report, confusion matrix, dan predictions_test.csv.",
        "note": "Macro F1 dan recall setiap kelas harus dibaca bersama confusion matrix, bukan accuracy saja.",
    },
    12: {
        "title": "Training baseline",
        "topic": "Melatih EdgeNeXt dengan konfigurasi awal.",
        "function": "Membuat hasil awal sebagai pembanding sebelum parameter tuning dilakukan.",
        "output": "Model baseline, metrik baseline, waktu training, dan penggunaan memory.",
        "note": "Baseline bukan model yang buruk; baseline adalah titik awal untuk mengukur pengaruh tuning.",
    },
    13: {
        "title": "Tuning EdgeNeXt",
        "topic": "Mencoba beberapa kombinasi learning rate, dropout, weight decay, dan label smoothing.",
        "function": "Mencari konfigurasi EdgeNeXt yang menghasilkan validation macro F1 terbaik.",
        "output": "Tabel tuning_results.csv dan best_config.",
        "note": "Pemilihan konfigurasi berdasarkan validation macro F1, bukan hanya accuracy.",
    },
    14: {
        "title": "Training model setelah tuning",
        "topic": "Membuat ulang EdgeNeXt dan melatihnya menggunakan konfigurasi terbaik.",
        "function": "Menghasilkan model final yang akan dipakai untuk evaluasi akhir dan pengujian eksternal.",
        "output": "Model final, history final, metrik resource, dan evaluasi test final.",
        "note": "Data test tetap sama dengan baseline agar perbandingan adil.",
    },
    15: {
        "title": "Perbandingan baseline dan tuning",
        "topic": "Membandingkan metrik performa, waktu, memory, parameter, dan ukuran model.",
        "function": "Menjawab apakah tuning benar-benar meningkatkan EdgeNeXt.",
        "output": "baseline_vs_tuning.csv dan kesimpulan otomatis tentang perubahan accuracy serta macro F1.",
        "note": "Tuning dianggap berhasil jika macro F1 meningkat tanpa trade-off yang tidak dapat diterima.",
    },
    16: {
        "title": "Contoh prediksi benar dan salah",
        "topic": "Menampilkan contoh gambar yang diprediksi benar dan salah oleh model final.",
        "function": "Membantu memahami jenis gambar yang mudah dan sulit dikenali model.",
        "output": "correct_examples.png dan wrong_examples.png.",
        "note": "Kesalahan prediksi perlu dibaca bersama confusion matrix untuk mengetahui pola kelas yang tertukar.",
    },
    17: {
        "title": "Grad-CAM",
        "topic": "Membuat heatmap untuk melihat area gambar yang paling memengaruhi keputusan model.",
        "function": "Memberikan interpretasi visual, bukan menggantikan diagnosis dokter.",
        "output": "gradcam_example.png dan nama layer convolution yang digunakan.",
        "note": "Heatmap yang baik seharusnya lebih fokus pada area anatomi atau tumor yang relevan, bukan latar belakang.",
    },
    18: {
        "title": "Menyimpan konfigurasi dan ringkasan",
        "topic": "Menyimpan seluruh konfigurasi, lokasi data, model, keterbatasan, dan ringkasan hasil.",
        "function": "Membuat eksperimen terdokumentasi dan mudah dilaporkan dalam skripsi.",
        "output": "experiment_config.json dan summary.txt.",
        "note": "BRISC2025 tidak menyediakan patient_id yang jelas, sehingga evaluasi utama berada pada tingkat gambar.",
    },
    19: {
        "title": "Audit overfitting",
        "topic": "Membandingkan training macro F1 dengan validation macro F1 pada epoch terbaik dan epoch terakhir.",
        "function": "Mencari indikasi bahwa model hanya menghafal data training.",
        "output": "overfitting_audit.csv dan kesimpulan gap training-validation.",
        "note": "Perbedaan training-validation perlu dibaca bersama metrik test internal.",
    },
}


def load_cell_sources() -> dict[int, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources: dict[int, str] = {}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        match = re.search(r"# Cell (\d+) -", source)
        if match:
            number = int(match.group(1))
            if 1 <= number <= 19:
                sources[number] = source.strip()
    missing = sorted(set(range(1, 20)) - set(sources))
    if missing:
        raise RuntimeError(f"Cell 1-19 tidak lengkap pada notebook: {missing}")
    return sources


def get_font(path: str, size: int):
    candidates = [
        Path(r"C:\Windows\Fonts") / path,
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_code_screenshot(cell_number: int, source: str) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    output = SCREENSHOT_DIR / f"cell_{cell_number:02d}.png"
    lines = source.splitlines()
    max_lines = 30
    if len(lines) > max_lines:
        shown = lines[:22] + ["...  [kode panjang dipersingkat pada PDF] ..."] + lines[-7:]
    else:
        shown = lines

    code_font = get_font("consola.ttf", 22)
    small_font = get_font("consola.ttf", 18)
    line_height = 30
    margin = 28
    width = 1900
    height = margin * 2 + 62 + len(shown) * line_height
    image = Image.new("RGB", (width, height), "#101820")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 56), fill="#263746")
    draw.ellipse((22, 20, 38, 36), fill="#ff5f56")
    draw.ellipse((48, 20, 64, 36), fill="#ffbd2e")
    draw.ellipse((74, 20, 90, 36), fill="#27c93f")
    draw.text((120, 16), f"Cell {cell_number} - pratinjau kode", font=small_font, fill="#e8eef2")

    y = 78
    for number, line in enumerate(shown, start=1):
        draw.text((margin, y), f"{number:02d}", font=small_font, fill="#718096")
        color = "#9cdcfe" if line.strip().startswith("#") else "#d4d4d4"
        if line.startswith("..."):
            color = "#ffd580"
        draw.text((100, y), line[:145], font=code_font, fill=color)
        y += line_height
    image.save(output)
    return output


def make_brain_grid() -> Path | None:
    """Membuat grid contoh MRI asli, satu gambar dari setiap kelas."""

    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
    selected = []
    for class_name in class_names:
        candidates = sorted((DATASET_DIR / "train" / class_name).glob("*.jpg"))
        if not candidates:
            continue
        selected.append((class_name, candidates[0]))
    if not selected:
        return None

    output = TMP_ROOT / "brain_examples_grid.png"
    font = get_font("segoeui.ttf", 30)
    label_font = get_font("segoeui.ttf", 25)
    tile_w, tile_h = 510, 570
    canvas = Image.new("RGB", (tile_w * len(selected), tile_h), "#f4f8fa")
    draw = ImageDraw.Draw(canvas)
    for index, (class_name, image_path) in enumerate(selected):
        with Image.open(image_path) as source:
            image = ImageOps.fit(source.convert("RGB"), (tile_w - 30, tile_w - 30))
        x = index * tile_w + 15
        canvas.paste(image, (x, 15))
        draw.text((x + 8, tile_w + 25), class_name, font=font, fill="#123047")
        draw.text((x + 8, tile_w + 69), "contoh MRI JPG 2D", font=label_font, fill="#5b6b77")
    canvas.save(output)
    return output


def make_flowchart_images() -> tuple[Path, Path]:
    """Membuat dua flowchart metodologi untuk dimasukkan ke PDF."""

    flow_dir = TMP_ROOT / "flowcharts"
    flow_dir.mkdir(parents=True, exist_ok=True)
    title_font = get_font("segoeui.ttf", 30)
    box_font = get_font("segoeui.ttf", 25)
    small_font = get_font("segoeui.ttf", 21)

    main_path = flow_dir / "flowchart_main.png"
    canvas = Image.new("RGB", (1500, 1850), "#f7fbfc")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 35), "Flowchart utama penelitian", font=title_font, fill="#123047")
    boxes = [
        ("1. Dataset BRISC2025", "JPG 2D, empat kelas", "#d9eff2"),
        ("2. Pemeriksaan data", "jumlah kelas, file rusak, duplikasi", "#e4f2e9"),
        ("3. Split data", "train 80%, validation 20%, test tetap", "#eaf0fb"),
        ("4. Preprocessing", "RGB, resize 224x224, normalisasi", "#f7ecd5"),
        ("5. DataLoader", "membuat batch untuk CPU/GPU", "#f1e2ef"),
        ("6. EdgeNeXt baseline", "belajar dengan konfigurasi awal", "#d9eff2"),
        ("7. Evaluasi baseline", "accuracy, macro F1, recall, confusion matrix", "#e4f2e9"),
        ("8. Tuning EdgeNeXt", "mencari learning rate, dropout, dan parameter terbaik", "#eaf0fb"),
        ("9. Model final", "training ulang memakai konfigurasi terbaik", "#f7ecd5"),
        ("10. Evaluasi dan laporan", "perbandingan, Grad-CAM, overfitting, data eksternal", "#f1e2ef"),
    ]
    x1, x2 = 130, 1370
    y = 115
    box_h = 135
    for index, (heading, detail, fill) in enumerate(boxes):
        draw.rounded_rectangle((x1, y, x2, y + box_h), radius=22, fill=fill, outline="#6c9eaa", width=3)
        draw.text((x1 + 28, y + 22), heading, font=box_font, fill="#123047")
        draw.text((x1 + 28, y + 72), detail, font=small_font, fill="#40566a")
        if index < len(boxes) - 1:
            draw.line((750, y + box_h, 750, y + box_h + 38), fill="#3f7180", width=5)
            draw.polygon([(737, y + box_h + 32), (763, y + box_h + 32), (750, y + box_h + 52)], fill="#3f7180")
        y += 165
    canvas.save(main_path)

    branch_path = flow_dir / "flowchart_training_branch.png"
    canvas = Image.new("RGB", (1800, 1100), "#f7fbfc")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 35), "Flowchart training, tuning, dan evaluasi", font=title_font, fill="#123047")
    nodes = {
        "start": (80, 430, 450, 610, "DataLoader\ntrain + validation"),
        "base": (600, 180, 1040, 360, "Baseline\nkonfigurasi awal"),
        "tune": (600, 680, 1040, 860, "Tuning terbatas\nvalidation macro F1"),
        "test_base": (1190, 180, 1660, 360, "Evaluasi baseline\nmodel dan resource"),
        "final": (1190, 680, 1660, 860, "Model final\nkonfigurasi terbaik"),
        "compare": (1190, 430, 1660, 610, "Perbandingan\nbaseline vs tuning"),
    }
    for fill, key in [("#d9eff2", "start"), ("#e4f2e9", "base"), ("#eaf0fb", "tune"), ("#f7ecd5", "test_base"), ("#f7ecd5", "final"), ("#f1e2ef", "compare")]:
        x1, y1, x2, y2, text = nodes[key]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=fill, outline="#6c9eaa", width=3)
        lines = text.split("\n")
        for line_index, line in enumerate(lines):
            draw.text((x1 + 30, y1 + 45 + line_index * 48), line, font=box_font, fill="#123047")

    def arrow(start, end):
        draw.line((start[0], start[1], end[0], end[1]), fill="#3f7180", width=6)
        import math
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        size = 22
        left = (end[0] - size * math.cos(angle - 0.5), end[1] - size * math.sin(angle - 0.5))
        right = (end[0] - size * math.cos(angle + 0.5), end[1] - size * math.sin(angle + 0.5))
        draw.polygon([end, left, right], fill="#3f7180")

    arrow((450, 480), (600, 270))
    arrow((450, 560), (600, 770))
    arrow((1040, 270), (1190, 270))
    arrow((1040, 770), (1190, 770))
    arrow((1425, 360), (1425, 430))
    arrow((1425, 680), (1425, 610))
    draw.text((520, 405), "training", font=small_font, fill="#5b6b77")
    draw.text((520, 875), "mencari konfigurasi", font=small_font, fill="#5b6b77")
    canvas.save(branch_path)
    return main_path, branch_path


def build_pdf() -> Path:
    sources = load_cell_sources()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=31, textColor=colors.HexColor("#123047"), alignment=TA_CENTER,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="CoverSub", parent=styles["Normal"], fontName="Helvetica",
        fontSize=12, leading=18, textColor=colors.HexColor("#40566a"), alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=23, textColor=colors.HexColor("#123047"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CellTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=17, leading=22, textColor=colors.HexColor("#123047"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodyID", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.6, leading=14, textColor=colors.HexColor("#243746"), spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SmallID", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8, leading=11, textColor=colors.HexColor("#5b6b77"),
    ))
    styles.add(ParagraphStyle(
        name="LabelID", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=9.5, leading=13, textColor=colors.HexColor("#0d5c75"), spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="FlowID", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=12, leading=19, textColor=colors.HexColor("#123047"), alignment=TA_CENTER,
    ))

    doc = BaseDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.5 * cm, bottomMargin=1.6 * cm,
        title="Panduan Alur Penelitian EdgeNeXt BRISC2025",
        author="Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")

    def draw_header_footer(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(colors.HexColor("#d6e2e8"))
        canvas.line(document.leftMargin, height - 1.15 * cm, width - document.rightMargin, height - 1.15 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(document.leftMargin, 0.85 * cm, "Panduan penelitian klasifikasi tumor otak dengan EdgeNeXt")
        canvas.drawRightString(width - document.rightMargin, 0.85 * cm, f"Halaman {document.page}")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="guide", frames=frame, onPage=draw_header_footer)])
    story = []

    # Cover
    story.append(Spacer(1, 2.0 * cm))
    story.append(Paragraph("Panduan Alur Penelitian", styles["CoverTitle"]))
    story.append(Paragraph("Klasifikasi Tumor Otak BRISC2025 menggunakan EdgeNeXt", styles["CoverTitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Penjelasan sederhana Cell 1 sampai Cell 19 untuk dipelajari dan dipresentasikan", styles["CoverSub"]))
    story.append(Spacer(1, 1.2 * cm))
    cover_table = Table([
        [Paragraph("Dataset", styles["LabelID"]), Paragraph("BRISC2025, gambar MRI T1 dalam JPG 2D", styles["BodyID"])],
        [Paragraph("Model", styles["LabelID"]), Paragraph("EdgeNeXt, satu model dengan perbandingan sebelum dan sesudah tuning", styles["BodyID"])],
        [Paragraph("Bahasa program", styles["LabelID"]), Paragraph("Python 3.10, PyTorch, torchvision, timm", styles["BodyID"])],
        [Paragraph("Tujuan", styles["LabelID"]), Paragraph("Mengukur performa, waktu, memory, generalisasi, dan interpretasi model", styles["BodyID"])],
    ], colWidths=[3.0 * cm, 12.2 * cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf6f8")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#96c5ce")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8dfe4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("Cara menggunakan PDF ini: baca alur besar terlebih dahulu, kemudian pelajari setiap cell sesuai urutan. Potongan kode ditampilkan sebagai pratinjau visual; kode lengkap tetap berada di notebook.", styles["BodyID"]))
    story.append(PageBreak())

    # Overview
    story.append(Paragraph("Alur besar penelitian", styles["SectionTitle"]))
    flow_lines = [
        "Dataset BRISC2025 JPG 2D",
        "-> Pemeriksaan dan pembersihan data",
        "-> Split train, validation, dan test",
        "-> Preprocessing dan DataLoader",
        "-> EdgeNeXt baseline",
        "-> Evaluasi baseline",
        "-> Tuning EdgeNeXt",
        "-> Training model final",
        "-> Evaluasi final dan perbandingan",
        "-> Grad-CAM dan audit overfitting",
        "-> Kesimpulan penelitian",
    ]
    flow_table = Table([[Paragraph("<br/>".join(flow_lines), styles["FlowID"])]], colWidths=[16.8 * cm])
    flow_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f8fb")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#79aebb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(flow_table)
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Inti penelitian: model yang dipakai tetap EdgeNeXt. Yang dibandingkan adalah EdgeNeXt dengan konfigurasi awal dan EdgeNeXt setelah tuning. Data test tidak boleh dipakai untuk memilih konfigurasi.", styles["BodyID"]))
    story.append(Paragraph("Catatan dataset: BRISC2025 sudah berupa JPG 2D. Karena tidak ada hubungan antar-slice dan patient_id yang jelas, penelitian utama dievaluasi pada tingkat gambar, bukan tingkat pasien.", styles["BodyID"]))
    story.append(PageBreak())

    flow_main, flow_branch = make_flowchart_images()
    story.append(Paragraph("Flowchart detail penelitian", styles["SectionTitle"]))
    story.append(Paragraph("Diagram ini menunjukkan urutan kerja dari data mentah sampai kesimpulan. Setiap kotak akan dijelaskan lebih detail pada Cell 1 sampai Cell 19.", styles["BodyID"]))
    main_img = Image.open(flow_main)
    main_width = 16.6 * cm
    main_height = main_width * main_img.size[1] / main_img.size[0]
    story.append(RLImage(str(flow_main), width=main_width, height=main_height))
    story.append(PageBreak())

    story.append(Paragraph("Flowchart training dan perbandingan", styles["SectionTitle"]))
    story.append(Paragraph("Baseline digunakan sebagai hasil awal. Tuning mencari konfigurasi yang lebih baik berdasarkan validation macro F1. Setelah itu model final diuji dengan data test yang sama.", styles["BodyID"]))
    branch_img = Image.open(flow_branch)
    branch_width = 16.6 * cm
    branch_height = branch_width * branch_img.size[1] / branch_img.size[0]
    story.append(RLImage(str(flow_branch), width=branch_width, height=branch_height))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Urutan pembacaan hasil: validation dipakai untuk memilih konfigurasi, test dipakai sekali untuk laporan akhir, sedangkan data eksternal dipakai untuk melihat generalisasi.", styles["BodyID"]))
    story.append(PageBreak())

    brain_grid = make_brain_grid()
    if brain_grid is not None or PREVIOUS_GRADCAM.exists():
        story.append(Paragraph("Contoh gambar otak dan keluaran visual", styles["SectionTitle"]))
        story.append(Paragraph("Gambar berikut berasal dari data dan hasil visual yang tersedia. Gambar MRI di bawah adalah contoh nyata dari BRISC2025, sedangkan Grad-CAM adalah contoh hasil run sebelumnya yang Anda kirimkan.", styles["BodyID"]))
        if brain_grid is not None:
            story.append(Paragraph("Contoh gambar MRI per kelas", styles["LabelID"]))
            grid_img = Image.open(brain_grid)
            grid_width = 16.6 * cm
            grid_height = grid_width * grid_img.size[1] / grid_img.size[0]
            story.append(RLImage(str(brain_grid), width=grid_width, height=grid_height))
            story.append(Spacer(1, 0.25 * cm))
        if PREVIOUS_GRADCAM.exists():
            story.append(Paragraph("Contoh Grad-CAM dari hasil sebelumnya", styles["LabelID"]))
            grad_img = Image.open(PREVIOUS_GRADCAM)
            grad_width = 16.6 * cm
            grad_height = grad_width * grad_img.size[1] / grad_img.size[0]
            story.append(RLImage(str(PREVIOUS_GRADCAM), width=grad_width, height=grad_height))
            story.append(Paragraph("Interpretasi: warna merah menunjukkan area yang paling memengaruhi prediksi model. Heatmap ini membantu memeriksa apakah model memperhatikan area otak yang relevan.", styles["SmallID"]))
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph("Catatan: hasil angka baru seperti accuracy, confusion matrix, dan Grad-CAM baru akan diperbarui setelah notebook dijalankan ulang. PDF ini tidak mengarang hasil training baru.", styles["BodyID"]))
        story.append(PageBreak())

    for number in range(1, 20):
        info = CELL_INFO[number]
        screenshot = make_code_screenshot(number, sources[number])
        story.append(Paragraph(f"Cell {number}: {info['title']}", styles["CellTitle"]))
        story.append(Paragraph(f"<b>Topik:</b> {info['topic']}", styles["BodyID"]))
        story.append(Paragraph(f"<b>Fungsi:</b> {info['function']}", styles["BodyID"]))
        story.append(Spacer(1, 0.12 * cm))
        image = Image.open(screenshot)
        width_px, height_px = image.size
        display_width = 16.6 * cm
        display_height = display_width * height_px / width_px
        story.append(RLImage(str(screenshot), width=display_width, height=display_height))
        story.append(Spacer(1, 0.16 * cm))
        detail_table = Table([
            [Paragraph("Hasil/output", styles["LabelID"]), Paragraph(info["output"], styles["BodyID"])],
            [Paragraph("Catatan", styles["LabelID"]), Paragraph(info["note"], styles["BodyID"])],
        ], colWidths=[3.0 * cm, 13.6 * cm])
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4f6")),
            ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#fbfdfe")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#a9cbd1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d4e5e8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(detail_table)
        if number in {5, 10, 15}:
            story.append(Spacer(1, 0.16 * cm))
            story.append(Paragraph("Hubungan dengan penelitian: cell ini adalah bagian penting dari alur dan hasilnya digunakan oleh cell-cell berikutnya.", styles["SmallID"]))
        story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Checklist sebelum Run All", styles["SectionTitle"]))
    checklist = [
        "Kernel VS Code menggunakan Python 3.10.",
        "DATA_DIR menunjuk ke folder classification_task yang benar.",
        "GPU/CUDA sudah terdeteksi jika ingin training dengan GPU.",
        "Folder hasil_edgenext boleh kosong agar hasil tidak tertumpuk.",
        "Cell dijalankan berurutan dari Cell 1 sampai Cell 19.",
        "Early stopping boleh menghentikan training sebelum epoch maksimum.",
        "Macro F1, recall, dan confusion matrix dibaca bersama accuracy.",
        "Hasil training tidak boleh langsung dianggap diagnosis medis.",
    ]
    story.append(Paragraph("<br/>".join(f"- {item}" for item in checklist), styles["BodyID"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("File sumber notebook: " + str(NOTEBOOK_PATH), styles["SmallID"]))
    doc.build(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    print(build_pdf())
