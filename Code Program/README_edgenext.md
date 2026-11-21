# BRISC2025 EdgeNeXt

Notebook penelitian:
- `notebooks/edgenext_brisc2025_classification.ipynb`

## Environment

- Python 3.10
- VS Code
- Python extension
- Jupyter extension

## Menjalankan notebook

1. Buka folder `E:\CNN(Tumor otak)\Code Program` di VS Code.
2. Pilih kernel Python 3.10 yang sudah memiliki library penelitian.
3. Buka notebook di folder `notebooks`.
4. Jalankan cell secara berurutan.
5. Hasil setiap run disimpan otomatis di:

```
hasil_edgenext/
└── YYYYMMDD_HHMMSS/
    ├── 01_baseline/
    ├── 02_tuning/
    ├── 03_final/
    └── 04_comparison/
```

Notebook mencari folder klasifikasi BRISC2025 pada dua lokasi berikut:

`E:\CNN(Tumor otak)\brisc2025-v6\brisc2025\classification_task`

atau:

`E:\CNN(Tumor otak)\Code Program\brisc2025-v6\brisc2025\classification_task`

Dataset test resmi tidak diubah. Folder train dibagi menjadi train dan validation secara stratified.

## Dataset eksternal Mendeley

Dataset eksternal baru diambil dari [Multi-Class Brain Tumor MRI Dataset](https://data.mendeley.com/datasets/82mtzd8x72/1), DOI `10.17632/82mtzd8x72.1`, dan disimpan di:

`E:\CNN(Tumor otak)\Code Program\external_datasets\mendeley_82mtzd8x72_v1`

Arsip lokal yang terdeteksi berisi 1.137 JPG: `Glioma` 295, `Healthy` 291, `Meningioma` 277, dan `Pituitary Macroadenoma` 274. Label `Healthy` dipetakan ke `no_tumor`, sedangkan `Pituitary Macroadenoma` dipetakan ke `pituitary`.

Halaman sumber menyebut empat kelas dan mencantumkan campuran sekuens T1/T2. Arsip yang diunduh tidak menyertakan metadata modality T1-kontras atau patient ID pada nama file. Karena itu, kode tidak menganggap seluruh arsip sebagai T1 dan tidak menjalankan evaluasi strict sebelum metadata tersebut diverifikasi.

Audit dan persiapan dijalankan dengan:

```powershell
python scripts\prepare_mendeley_external.py `
  --external-root external_datasets\mendeley_82mtzd8x72_v1\raw `
  --brisc-manifest brisc2025-v6\brisc2025\manifest.csv `
  --output-dir hasil_edgenext\external_mendeley_audit
```

Script menghitung SHA-256 dan pHash, melaporkan overlap per dataset/per kelas, menandai nama file yang mengindikasikan augmentasi, membuat preprocessing crop-pad-resize dan normalisasi intensitas, serta menolak GroupKFold jika patient/group ID belum tersedia. Template verifikasi dibuat sebagai `verified_metadata_template.csv`; kolom `modality_status` harus diisi `t1_verified` hanya berdasarkan metadata sumber yang benar-benar dapat dibuktikan.

Ukuran yang dijamin adalah ukuran input setelah preprocessing, bukan ukuran file JPG mentah. Cell 8 memeriksa setiap item yang masuk `BrainTumorDataset`, sedangkan Cell 20 memeriksa seluruh gambar internal dan eksternal yang diaudit. Setiap tensor harus berbentuk `3 x 224 x 224`; jika satu gambar berbeda, cell melempar error dan menghentikan proses.

## Catatan penelitian

Dataset yang digunakan sudah berupa gambar JPG 2D. Notebook tidak melakukan konversi volume 3D atau penggabungan 2.5D. Karena patient_id tidak tersedia dengan jelas, evaluasi dilakukan pada tingkat gambar.

## Audit hasil tersimpan

Run historis `hasil_edgenext/20261003_195329` memperoleh accuracy 99.2% pada test BRISC2025 (1.000 gambar). Nilai itu dibuat sebelum pipeline crop-pad-normalisasi terbaru dan harus dianggap sebagai baseline historis, bukan hasil final setelah perubahan. Pemisahan train-validation BRISC dilakukan per gambar, bukan per pasien, sehingga nilai ini belum menjamin generalisasi ke pasien baru.

Kode notebook mengatur seed yang sama untuk setiap eksperimen, mencatat pemakaian RAM proses, dan membedakan waktu tuning dari waktu fit model akhir. Jalankan ulang notebook berurutan untuk membuat hasil baru yang sesuai dengan perubahan tersebut.
