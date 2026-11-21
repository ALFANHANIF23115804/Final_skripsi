# Penjelasan Lengkap Cell 1 sampai Cell 20 - EdgeNeXt BRISC2025

Dokumen ini menjelaskan notebook `edgenext_brisc2025_classification.ipynb` secara berurutan berdasarkan **cell kode bernomor Cell 1 sampai Cell 20**. Notebook ini membangun eksperimen klasifikasi tumor otak dari dataset BRISC2025 menggunakan model **EdgeNeXt** dari library `timm`.

Ringkasnya, alur notebook adalah:

1. menyiapkan library dan environment,
2. menentukan konfigurasi eksperimen,
3. membaca dataset dan membersihkan duplikasi,
4. membuat visualisasi dataset,
5. membagi data train/validation/test,
6. membuat transformasi gambar dan DataLoader,
7. membuat model EdgeNeXt,
8. melatih baseline,
9. melakukan tuning,
10. melatih ulang model final,
11. mengevaluasi performa,
12. membuat visualisasi prediksi dan Grad-CAM,
13. menyimpan ringkasan eksperimen,
14. melakukan audit overfitting,
15. menyiapkan audit dataset eksternal Mendeley dan memeriksa ukuran tensor 224.

## Cell 1 - Import Library dan Pemeriksaan Environment

Cell ini berfungsi sebagai fondasi awal notebook. Semua library penting dipanggil di sini agar cell berikutnya bisa memakai fungsi, class, dan modul yang dibutuhkan.

Bagian import standar Python:

- `gc`: membersihkan objek dari memori secara manual, terutama setelah tuning beberapa model.
- `hashlib`: membuat hash SHA-256 untuk mendeteksi file gambar yang identik.
- `json`: menyimpan konfigurasi dan metrik ke file `.json`.
- `os`, `platform`, `sys`: membaca informasi sistem, path Python, dan environment.
- `random`: mengontrol random seed Python.
- `re`: memproses nama file dengan regular expression, misalnya mendeteksi bidang MRI dari nama file.
- `time`: membuat `RUN_ID`, menghitung durasi training, dan durasi inferensi.
- `warnings`: menyembunyikan warning yang tidak penting agar output notebook lebih bersih.
- `Path` dari `pathlib`: mengelola path folder/file dengan cara yang lebih aman dan rapi.

Bagian library data dan visualisasi:

- `matplotlib.pyplot as plt`: membuat grafik, menyimpan gambar distribusi, kurva training, confusion matrix, dan Grad-CAM.
- `numpy as np`: operasi numerik, array, dan perhitungan metrik.
- `pandas as pd`: membuat tabel manifest dataset, laporan kualitas data, history training, dan hasil prediksi.
- `psutil`: membaca penggunaan RAM.
- `seaborn as sns`: membuat visualisasi statistik seperti countplot dan heatmap confusion matrix.

Bagian machine learning:

- `sklearn`: dicek versinya.
- `accuracy_score`: menghitung akurasi.
- `classification_report`: menghasilkan precision, recall, F1-score per kelas.
- `confusion_matrix`: membuat matriks prediksi benar/salah antar kelas.
- `precision_recall_fscore_support`: menghitung precision, recall, dan macro F1.
- `roc_auc_score`: menghitung ROC-AUC multi-class.
- `train_test_split`: membagi train menjadi train dan validation.
- `label_binarize`: mengubah label kelas menjadi bentuk one-hot untuk ROC-AUC.

Bagian PyTorch:

- `torch`: framework utama deep learning.
- `torch.nn as nn`: modul neural network seperti `CrossEntropyLoss`.
- `torchvision`: dicek versinya.
- `DataLoader` dan `Dataset`: membuat pipeline data mini-batch.
- `transforms`: resize, augmentasi, konversi tensor, dan normalisasi gambar.

Bagian gambar:

- `Image` dan `ImageFile` dari PIL: membuka, memverifikasi, dan mengubah gambar MRI menjadi RGB.
- `ImageFile.LOAD_TRUNCATED_IMAGES = False`: gambar yang terpotong/rusak tidak dipaksa dibaca. Ini penting agar data rusak bisa ketahuan.

Bagian `timm`:

- Notebook mencoba `import timm`.
- Kalau `timm` belum terpasang, variabel `timm` diisi `None`.
- `timm` diperlukan karena model EdgeNeXt dibuat dari `timm.create_model`.

Bagian output environment:

- Mencetak versi Python, path Python, versi PyTorch, Torchvision, Scikit-learn, versi `timm`, build CUDA, dan status CUDA.
- Jika CUDA tersedia, notebook mencetak nama GPU dan memori GPU awal.
- Jika CUDA tidak tersedia, notebook memberi catatan bahwa training akan berjalan di CPU.

Fungsi cell ini: memastikan semua library tersedia dan memberi bukti environment yang dipakai, sehingga hasil eksperimen lebih mudah dilacak.

## Cell 2 - Konfigurasi Penelitian

Cell ini menentukan seluruh konfigurasi eksperimen.

`PROJECT_ROOT` diarahkan ke folder proyek:

```python
PROJECT_ROOT = Path(r"E:\CNN(Tumor otak)\Code Program")
```

`DATA_DIR_CANDIDATES` berisi dua kemungkinan lokasi dataset `classification_task`. Notebook lalu memilih path pertama yang benar-benar ada. Ini membuat notebook lebih tahan terhadap perbedaan struktur folder.

Jika dataset tidak ditemukan, cell akan menghentikan eksekusi dengan `FileNotFoundError`. Pesan error-nya juga menyebut lokasi yang sudah diperiksa.

Folder output:

- `OUTPUT_ROOT`: folder utama hasil eksperimen, yaitu `hasil_edgenext`.
- `RUN_ID`: timestamp waktu eksperimen, misalnya `20261003_195329`.
- `RUN_ROOT`: folder khusus untuk satu run eksperimen.
- `BASELINE_DIR`: menyimpan hasil model baseline.
- `TUNING_DIR`: menyimpan hasil tuning.
- `FINAL_DIR`: menyimpan hasil model final setelah tuning.
- `COMPARISON_DIR`: menyimpan hasil perbandingan, Grad-CAM, contoh prediksi, dan ringkasan.

Cell ini membuat semua folder output dengan:

```python
folder.mkdir(parents=True, exist_ok=True)
```

Konfigurasi dataset dan model:

- `CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]`: empat kelas klasifikasi tumor otak.
- `NUM_CLASSES = 4`: jumlah kelas.
- `IMAGE_SIZE = 224`: semua gambar diubah ke 224x224 piksel.
- `BATCH_SIZE = 16`: setiap batch berisi 16 gambar.
- `NUM_WORKERS = 0`: DataLoader berjalan di proses utama. Ini sering lebih aman di Windows/Jupyter.
- `LEARNING_RATE = 3e-4`: learning rate awal baseline.
- `WEIGHT_DECAY = 1e-4`: regularisasi optimizer baseline.
- `VAL_SIZE = 0.20`: 20% data train dipakai sebagai validation.
- `RANDOM_SEED = 42`: membuat pembagian data dan random process lebih reproducible.
- `MODEL_NAME = "edgenext_xx_small"`: varian EdgeNeXt yang dipakai.
- `USE_PRETRAINED = True`: memakai bobot pretrained.
- `BASELINE_EPOCHS`, `TUNING_EPOCHS`, `FINAL_EPOCHS`: masing-masing maksimal 30 epoch.
- `EARLY_STOPPING_PATIENCE = 5`: training berhenti jika validation macro F1 tidak membaik selama 5 epoch.
- `MAX_TUNING_TRIALS = 2`: tuning hanya mencoba 2 konfigurasi.

Konfigurasi device:

- `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Jika CUDA aktif, model dan tensor akan memakai GPU.
- Jika tidak, semuanya memakai CPU.
- `PIN_MEMORY` aktif hanya saat memakai CUDA.

Fungsi cell ini: menjadi pusat kontrol eksperimen. Jika ingin mengubah ukuran gambar, batch size, model, jumlah epoch, atau lokasi output, tempatnya ada di cell ini.

## Cell 3 - Fungsi Bantuan

Cell ini membuat fungsi utilitas yang dipakai banyak cell berikutnya.

`IMAGE_EXTENSIONS` menentukan ekstensi gambar yang dianggap valid:

```python
{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
```

`seed_everything(seed=RANDOM_SEED)`:

- mengatur seed untuk `random`,
- mengatur seed untuk NumPy,
- mengatur seed untuk PyTorch CPU,
- mengatur seed untuk PyTorch CUDA jika GPU tersedia,
- membuat CuDNN deterministic,
- mematikan benchmark CuDNN agar hasil lebih stabil.

Tujuannya adalah reproducibility: pembagian data, augmentasi, dan training lebih konsisten antar run.

`gpu_memory_mb()`:

- membaca peak GPU memory dari PyTorch,
- mengembalikan 0 jika CUDA tidak tersedia.

`ram_used_mb()`:

- membaca RAM sistem yang sedang digunakan memakai `psutil.virtual_memory().used`.

`save_json(data, path)`:

- menyimpan dictionary/list ke file JSON.
- Di dalamnya ada fungsi `convert(value)` agar objek yang tidak otomatis bisa disimpan ke JSON dapat dikonversi.
- Contoh objek yang dikonversi: `Path`, `torch.device`, `torch.dtype`, `np.ndarray`, `np.integer`, `np.floating`, `np.bool_`, list, tuple, dan dict.

Di akhir cell, `seed_everything()` dipanggil langsung agar seluruh proses setelahnya memakai seed yang sudah ditentukan.

Fungsi cell ini: menyediakan alat bantu untuk reproducibility, monitoring resource, dan penyimpanan JSON.

## Cell 4 - Membaca Manifest Train dan Test

Cell ini membangun daftar semua gambar train dan test, lalu membersihkan duplikasi.

`detect_plane(filename)`:

- membaca nama file dalam huruf kecil,
- memecah nama file berdasarkan `_`, `-`, dan `.`,
- jika menemukan `axial` atau token `ax`, hasilnya `axial`,
- jika menemukan `coronal` atau token `co`, hasilnya `coronal`,
- jika menemukan `sagittal` atau token `sa`, hasilnya `sagittal`,
- jika tidak terdeteksi, hasilnya `unknown`.

Fungsi ini berguna untuk melihat distribusi bidang MRI berdasarkan nama file.

`file_sha256(image_path)`:

- membaca file gambar dalam bentuk byte,
- menghitung hash SHA-256,
- hasil hash dipakai sebagai sidik jari file.

Jika dua file punya hash sama, berarti isi file-nya identik.

`scan_split(split_name)`:

- menerima nama split, misalnya `"train"` atau `"test"`,
- mencari folder `DATA_DIR / split_name`,
- memastikan folder split ada,
- untuk setiap kelas di `CLASS_NAMES`, memastikan folder kelas ada,
- membaca semua file gambar valid di folder kelas,
- membuat record berisi:
  - `image_path`,
  - `split`,
  - `label`,
  - `label_id`,
  - `filename`,
  - `plane`,
  - `file_sha256`.
- mengembalikan hasil sebagai DataFrame.

`validate_manifest_paths(dataframe, split_name, verify_images=False)`:

- memastikan semua path gambar benar-benar ada,
- jika ada file hilang, daftar file disimpan ke `missing_files_<split>.txt` dan eksekusi dihentikan,
- jika `verify_images=True`, setiap gambar dibuka dengan PIL dan diverifikasi,
- jika ada gambar rusak, daftar disimpan ke `corrupt_files_<split>.csv` dan eksekusi dihentikan.

Setelah itu:

- `train_all_df = scan_split("train")`
- `test_df = scan_split("test")`
- keduanya divalidasi dengan `verify_images=True`.

Pembersihan data leakage:

- `test_hashes` berisi hash semua gambar test.
- `train_test_duplicate_df` mencari gambar train yang identik dengan gambar test.
- gambar train yang identik dengan test dikeluarkan dari train.
- `train_duplicate_df` mencari duplikasi internal di train.
- duplikasi internal train juga dibuang.
- test tidak diubah.

Ini penting karena jika gambar yang sama muncul di train dan test, metrik test bisa terlihat terlalu tinggi.

Cell ini membuat `data_quality_report.csv` yang berisi:

- jumlah train original,
- jumlah duplikasi train-test yang dihapus dari train,
- jumlah duplikasi internal train yang dihapus,
- jumlah train setelah cleanup,
- jumlah test asli yang tidak diubah.

Output penting dari run terbaru:

- duplikasi train-test yang dikeluarkan dari train: 9,
- duplikasi internal train yang dikeluarkan: 36,
- jumlah train setelah pembersihan: 4955,
- jumlah test: 1000.

Cell ini juga menampilkan tabel distribusi kelas dan distribusi bidang MRI.

Fungsi cell ini: membangun manifest dataset yang bersih, memverifikasi file, dan mencegah data leakage berbasis file identik.

## Cell 5 - Grafik Distribusi dan Pemeriksaan File Rusak

Cell ini menghasilkan gambar distribusi kelas dan melakukan pengecekan ulang file rusak.

`distribution` dibuat dengan menggabungkan:

- `train_all_df` dengan kolom `split="train"`,
- `test_df` dengan kolom `split="test"`.

Visualisasi:

```python
sns.countplot(data=distribution, x="label", hue="split", order=CLASS_NAMES)
```

Grafik ini memperlihatkan jumlah gambar tiap kelas pada train dan test.

Gambar disimpan ke:

```text
hasil_edgenext/<RUN_ID>/dataset_distribution.png
```

`find_corrupt_images(df)`:

- membuka semua gambar dalam DataFrame,
- menjalankan `image.verify()`,
- jika gagal, path dan pesan error dimasukkan ke DataFrame.

Cell ini menghitung:

- `corrupt_train_df`,
- `corrupt_test_df`,
- `corrupt_df`.

Jika ada file rusak, hasilnya ditampilkan dan disimpan ke `corrupt_images.csv`.

Output run terbaru:

- file rusak train: 0,
- file rusak test: 0.

Fungsi gambar `dataset_distribution.png`:

- mengecek apakah kelas dataset seimbang atau timpang,
- membandingkan jumlah train dan test per kelas,
- membantu membaca apakah metrik model bisa bias karena distribusi kelas.

## Cell 6 - Menampilkan Contoh Gambar Setiap Kelas

Cell ini membuat grid contoh gambar dari tiap kelas.

Kode:

```python
fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(12, 12))
```

Artinya:

- jumlah baris = jumlah kelas, yaitu 4,
- jumlah kolom = 4 contoh gambar,
- total maksimal gambar = 16.

Untuk setiap kelas:

- mengambil 4 gambar pertama dari `train_all_df`,
- membuka gambar dengan PIL,
- mengubah gambar ke RGB,
- menampilkan gambar dengan `ax.imshow(image)`,
- memberi judul sesuai nama kelas.

Gambar disimpan ke:

```text
hasil_edgenext/<RUN_ID>/sample_images.png
```

Fungsi gambar `sample_images.png`:

- memastikan gambar benar-benar terbaca,
- mengecek visual tiap kelas secara manusiawi,
- melihat apakah gambar MRI punya format yang masuk akal,
- membantu mendeteksi masalah awal seperti label tertukar, gambar kosong, atau gambar terlalu berbeda.

## Cell 7 - Stratified Split

Cell ini membagi data train yang sudah dibersihkan menjadi train dan validation.

Kode utama:

```python
train_test_split(
    train_all_df,
    test_size=VAL_SIZE,
    stratify=train_all_df["label"],
    random_state=RANDOM_SEED,
)
```

Maknanya:

- `train_all_df` dibagi menjadi `train_df` dan `val_df`,
- `test_size=0.20`, jadi 20% menjadi validation,
- `stratify=train_all_df["label"]` menjaga proporsi kelas agar train dan validation tetap seimbang,
- `random_state=42` membuat split bisa direproduksi.

Setelah split:

- index train, validation, dan test di-reset,
- jumlah data ditampilkan,
- tabel distribusi kelas untuk train, validation, dan test dibuat,
- tabel disimpan ke `split_summary.csv`.

Output run terbaru:

- train: 3964,
- validation: 991,
- test: 1000.

Fungsi cell ini: menyediakan data validation untuk memilih model terbaik dan early stopping, tanpa menyentuh test set.

## Cell 8 - Dataset, Transform, dan DataLoader

Cell ini mengubah DataFrame gambar menjadi objek PyTorch Dataset dan DataLoader.

Normalisasi ImageNet:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

Ini dipakai karena model pretrained umumnya dilatih dengan normalisasi ImageNet.

`make_transforms(image_size, training=False)`:

Jika `training=True`, transformasi berisi:

- crop area foreground/otak secara heuristik,
- padding menjadi persegi,
- resize ke 224x224,
- random resized crop ringan,
- horizontal flip,
- perubahan gamma ringan,
- color jitter brightness dan contrast 10%,
- Gaussian blur ringan,
- konversi ke tensor,
- noise Gaussian ringan,
- normalisasi ImageNet.

Jika `training=False`, transformasi hanya:

- crop dan pad area otak,
- normalisasi intensitas per gambar pada percentile 1 sampai 99,
- resize 224x224,
- tensor,
- normalisasi.

Augmentasi hanya dipakai untuk train, bukan validation/test, agar evaluasi stabil.

`crop_brain_pad_square(image)` mengurangi border hitam dominan, lalu menambahkan padding agar rasio aspek tidak berubah. `preprocess_brain_image(image, image_size)` menjalankan crop, padding, resize, dan normalisasi intensitas per gambar sebelum tensor dibuat. `RandomGamma` meniru variasi intensitas scanner secara ringan. `AddGaussianNoise` menambahkan noise kecil hanya pada train.

`BrainTumorDataset(Dataset)`:

- menyimpan DataFrame,
- menyimpan transform,
- memeriksa apakah semua `image_path` ada,
- `__len__` mengembalikan jumlah data,
- `__getitem__` membuka satu gambar, mengubah ke RGB, menerapkan transform, lalu mengembalikan:
  - tensor gambar,
  - `label_id`,
  - path gambar.

`make_loaders()`:

- memvalidasi ulang path train, validation, dan test,
- membuat `BrainTumorDataset` untuk masing-masing split,
- membuat `DataLoader`:
  - train memakai `shuffle=True`,
  - validation dan test memakai `shuffle=False`.

Di akhir cell:

```python
images, labels, paths = next(iter(train_loader))
```

Ini mengambil satu batch contoh untuk memastikan DataLoader bekerja.

Output run terbaru:

- bentuk batch gambar: `torch.Size([16, 3, 224, 224])`,
- bentuk label: `torch.Size([16])`.

Makna bentuk gambar:

- 16 = jumlah gambar per batch,
- 3 = channel RGB,
- 224 = tinggi,
- 224 = lebar.

Fungsi cell ini: membuat pipeline data siap training untuk PyTorch.

## Cell 9 - Membuat Model EdgeNeXt

Cell ini membuat model EdgeNeXt dari library `timm`.

`create_edgenext(...)`:

- memastikan `timm` sudah tersedia,
- mengambil daftar model yang mengandung nama `edgenext`,
- memeriksa apakah `MODEL_NAME` ada di daftar tersebut,
- membuat model dengan:
  - `model_name`,
  - `pretrained`,
  - `num_classes`,
  - `drop_rate`.

`num_classes=4` membuat classifier akhir model disesuaikan dengan empat kelas tumor.

Blok `try/except TypeError` dibuat karena beberapa versi `timm` mungkin tidak menerima argumen `drop_rate`. Jika error, model dibuat tanpa `drop_rate`.

`count_parameters(model)`:

- menghitung total parameter,
- menghitung parameter yang trainable.

`model_size_mb(model_path)`:

- menghitung ukuran file model dalam MB.

Jika `timm` tersedia, notebook menampilkan:

- daftar model EdgeNeXt yang tersedia,
- membuat preview model,
- menghitung parameter,
- menghapus preview model dari memori.

Output run terbaru:

- model tersedia: `edgenext_base`, `edgenext_small`, `edgenext_small_rw`, `edgenext_x_small`, `edgenext_xx_small`,
- total parameter: 1,158,892,
- parameter trainable: 1,158,892.

Fungsi cell ini: memastikan arsitektur EdgeNeXt siap dipakai untuk klasifikasi 4 kelas.

## Cell 10 - Fungsi Training

Cell ini mendefinisikan fungsi training utama.

`run_epoch(model, loader, criterion, optimizer=None)` menjalankan satu epoch.

Jika `optimizer` diberikan:

- mode training aktif,
- gradient dihitung,
- loss di-backpropagation,
- parameter model diperbarui.

Jika `optimizer=None`:

- mode evaluasi/validation,
- gradient tidak dihitung,
- parameter model tidak berubah.

Langkah di dalam loop batch:

1. gambar dan label dipindahkan ke `DEVICE`,
2. optimizer di-reset dengan `zero_grad`,
3. model menghasilkan `logits`,
4. loss dihitung dengan `criterion`,
5. jika training, `loss.backward()` dan `optimizer.step()` dijalankan,
6. loss batch ditambahkan ke total loss,
7. target asli dan prediksi disimpan untuk menghitung metrik.

Metrik yang dikembalikan:

- average loss,
- accuracy,
- macro F1.

Macro F1 penting karena menghitung rata-rata F1 semua kelas tanpa memberi bobot lebih ke kelas yang jumlahnya besar.

`fit_model(model, train_loader, val_loader, config, save_dir)` melatih model sampai selesai.

Komponen penting:

- `CrossEntropyLoss` untuk klasifikasi multi-class.
- `label_smoothing` bisa dipakai untuk mencegah model terlalu percaya diri.
- `AdamW` sebagai optimizer.
- `ReduceLROnPlateau` menurunkan learning rate jika validation macro F1 tidak membaik.
- `early stopping` menghentikan training jika validation macro F1 tidak membaik selama beberapa epoch.

Selama training, setiap epoch menyimpan:

- epoch,
- train loss,
- train accuracy,
- train macro F1,
- validation loss,
- validation accuracy,
- validation macro F1,
- learning rate.

Best model dipilih berdasarkan validation macro F1, bukan loss atau accuracy. Bobot terbaik disimpan di `best_state`.

Setelah training:

- model dikembalikan ke bobot terbaik,
- model disimpan sebagai `best_model.pt`,
- history disimpan sebagai `history.csv`,
- resource metrics disimpan sebagai `resource_metrics.json`,
- config disimpan sebagai `config.json`,
- dibuat grafik loss,
- dibuat grafik accuracy dan macro F1.

Gambar yang dihasilkan:

- `loss_curve.png`: membandingkan train loss dan validation loss per epoch.
- `metric_curve.png`: menampilkan train accuracy, validation accuracy, dan validation macro F1 per epoch.

Fungsi gambar `loss_curve.png`:

- melihat apakah loss turun,
- melihat apakah validation loss mulai naik saat train loss turun, yang bisa menjadi tanda overfitting.

Fungsi gambar `metric_curve.png`:

- melihat perkembangan akurasi dan macro F1,
- melihat epoch terbaik,
- melihat stabilitas validation performance.

Fungsi cell ini: menjadi mesin training untuk baseline, tuning, dan final model.

## Cell 11 - Fungsi Evaluasi Test

Cell ini membuat fungsi evaluasi model pada data test atau data eksternal berlabel.

Decorator:

```python
@torch.no_grad()
```

Artinya PyTorch tidak menyimpan gradient. Evaluasi jadi lebih hemat memori dan lebih cepat.

`evaluate_model(model, loader, split_name, save_dir)`:

1. model masuk mode evaluasi dengan `model.eval()`,
2. waktu mulai inferensi dicatat,
3. setiap batch gambar masuk ke model,
4. logits diubah menjadi probabilitas dengan `softmax`,
5. prediksi diambil dari probabilitas tertinggi,
6. label asli, prediksi, probabilitas, dan path gambar dikumpulkan.

Metrik yang dihitung:

- accuracy,
- precision macro,
- recall macro,
- macro F1,
- total inference seconds,
- inference milliseconds per image,
- ROC-AUC OVR macro jika bisa dihitung.

`classification_report` menghasilkan tabel per kelas:

- precision,
- recall,
- F1-score,
- support.

`confusion_matrix` dibuat dan divisualisasikan dengan heatmap.

File yang disimpan:

- `classification_report_<split>.csv`,
- `confusion_matrix_<split>.png`,
- `predictions_<split>.csv`,
- `metrics_<split>.json`.

Kolom `predictions_df`:

- `image_path`,
- `true_label`,
- `predicted_label`,
- `correct`,
- `confidence`.

Fungsi gambar `confusion_matrix_test.png`:

- baris menunjukkan label sebenarnya,
- kolom menunjukkan prediksi model,
- angka diagonal menunjukkan prediksi benar,
- angka di luar diagonal menunjukkan kesalahan,
- membantu melihat kelas mana yang sering tertukar.

Fungsi cell ini: mengevaluasi model secara lengkap dan menyimpan hasil numerik maupun visual.

## Cell 12 - Training Baseline

Cell ini melatih model baseline, yaitu model awal sebelum tuning.

`baseline_config` berisi:

- `model_name`: `edgenext_xx_small`,
- `pretrained`: `True`,
- `drop_rate`: 0.0,
- `label_smoothing`: 0.0,
- `image_size`: 224,
- `batch_size`: 16,
- `epochs`: 30,
- `learning_rate`: 3e-4,
- `weight_decay`: 1e-4,
- `early_stopping_patience`: 5.

Jika `timm` tidak tersedia, training dihentikan.

Alur:

1. membuat `baseline_model`,
2. melatih model dengan `fit_model`,
3. mengevaluasi model pada test set dengan `evaluate_model`.

Output visual dari cell ini:

- `01_baseline/loss_curve.png`,
- `01_baseline/metric_curve.png`,
- `01_baseline/confusion_matrix_test.png`.

Output metrik run terbaru:

- accuracy baseline: 0.987,
- precision macro: 0.9894,
- recall macro: 0.9882,
- macro F1: 0.9887,
- ROC-AUC OVR macro: 0.9989,
- waktu training: sekitar 1920.74 detik,
- waktu inferensi: sekitar 19.98 detik,
- ukuran model: sekitar 4.4763 MB.

Fungsi cell ini: menyediakan performa pembanding sebelum parameter model dituning.

## Cell 13 - Tuning Terbatas

Cell ini mencoba beberapa konfigurasi hyperparameter untuk mencari performa validation terbaik.

`tuning_space` berisi dua konfigurasi:

Trial 1:

- `drop_rate`: 0.20,
- `learning_rate`: 1e-4,
- `weight_decay`: 5e-4,
- `label_smoothing`: 0.10.

Trial 2:

- `drop_rate`: 0.30,
- `learning_rate`: 3e-4,
- `weight_decay`: 5e-4,
- `label_smoothing`: 0.05.

`[:MAX_TUNING_TRIALS]` membatasi jumlah trial sesuai konfigurasi. Pada notebook ini maksimal 2 trial.

Untuk setiap trial:

1. membuat folder `trial_01`, `trial_02`, dan seterusnya,
2. membuat DataLoader sesuai config trial,
3. membuat model EdgeNeXt baru,
4. melatih model dengan `fit_model`,
5. menyimpan hasil ke `tuning_results`,
6. menyimpan ringkasan ke `tuning_rows`.

Setelah semua trial:

- dibuat `tuning_df`,
- diurutkan berdasarkan `best_validation_macro_f1`,
- disimpan sebagai `tuning_results.csv`,
- trial terbaik dipilih sebagai `best_config`.

Output run terbaru:

- trial terbaik: trial 2,
- best validation macro F1 trial 2: 0.9882,
- trial 2 memakai `drop_rate=0.3`, `learning_rate=0.0003`, `weight_decay=0.0005`, `label_smoothing=0.05`.

Gambar yang dihasilkan:

- setiap trial menghasilkan `loss_curve.png`,
- setiap trial menghasilkan `metric_curve.png`.

Fungsi cell ini: mencari kombinasi hyperparameter yang lebih baik daripada baseline berdasarkan validation macro F1.

## Cell 14 - Training Model Setelah Tuning

Cell ini melatih model final menggunakan konfigurasi terbaik dari Cell 13.

`final_config` dibuat dari:

- `best_config`,
- `epochs = FINAL_EPOCHS`,
- `early_stopping_patience = EARLY_STOPPING_PATIENCE`.

Kemudian:

1. DataLoader dibuat ulang berdasarkan `final_config`,
2. model EdgeNeXt baru dibuat dengan `drop_rate` dan pretrained sesuai `final_config`,
3. model dilatih dengan `fit_model`,
4. model final dievaluasi pada test set dengan `evaluate_model`.

Output visual:

- `03_final/loss_curve.png`,
- `03_final/metric_curve.png`,
- `03_final/confusion_matrix_test.png`.

Output run terbaru:

- accuracy final: 0.992,
- precision macro: 0.9933,
- recall macro: 0.9928,
- macro F1: 0.9930,
- ROC-AUC OVR macro: 0.9990,
- waktu training: sekitar 1242.15 detik,
- waktu inferensi: sekitar 10.72 detik,
- ukuran model: sekitar 4.4763 MB.

Fungsi cell ini: menghasilkan model akhir yang dipakai untuk laporan, prediksi contoh, Grad-CAM, dan pengujian eksternal.

## Cell 15 - Tabel Perbandingan Otomatis

Cell ini membandingkan baseline dan model final setelah tuning.

Data yang dibandingkan:

- accuracy,
- precision macro,
- recall macro,
- macro F1-score,
- ROC-AUC OVR macro,
- waktu training,
- waktu inferensi,
- waktu inferensi per gambar,
- peak GPU memory,
- RAM setelah training,
- jumlah parameter,
- ukuran model.

Hasil disimpan ke:

```text
04_comparison/baseline_vs_tuning.csv
```

Output run terbaru:

- accuracy naik dari 0.987 menjadi 0.992,
- macro F1 naik dari 0.9887 menjadi 0.9930,
- tuning dinyatakan berhasil berdasarkan macro F1,
- waktu inferensi final lebih rendah daripada baseline,
- jumlah parameter tetap sama karena arsitektur model tidak berubah.

Cell ini juga menghitung:

- `delta_accuracy`,
- `delta_f1`,
- kelas tersulit baseline berdasarkan recall terendah,
- kelas tersulit final berdasarkan recall terendah.

Fungsi cell ini: memberi kesimpulan otomatis apakah tuning meningkatkan performa atau tidak.

## Cell 16 - Contoh Prediksi Benar dan Salah

Cell ini menampilkan contoh gambar yang diprediksi benar dan contoh yang diprediksi salah oleh model final.

`show_prediction_examples(predictions_df, title, correct=True, n=8)`:

- memilih baris dari `predictions_df` berdasarkan kolom `correct`,
- jika `correct=True`, mengambil prediksi benar,
- jika `correct=False`, mengambil prediksi salah,
- maksimal mengambil 8 gambar,
- membuat grid gambar,
- menampilkan:
  - label asli,
  - label prediksi,
  - confidence.

Gambar yang dihasilkan:

- `correct_examples.png`,
- `wrong_examples.png`.

Fungsi gambar `correct_examples.png`:

- memperlihatkan contoh kasus yang berhasil dikenali model,
- membantu melihat pola gambar yang mudah bagi model.

Fungsi gambar `wrong_examples.png`:

- memperlihatkan kasus salah prediksi,
- sangat penting untuk analisis error,
- membantu melihat kelas mana yang tertukar dan apakah kesalahan tampak wajar secara visual.

Catatan penting:

- `confidence` adalah keyakinan model berdasarkan softmax.
- Confidence tinggi bukan jaminan diagnosis benar.
- Untuk konteks medis, hasil tetap perlu divalidasi ahli.

## Cell 17 - Grad-CAM Sederhana

Cell ini mencoba menjelaskan area gambar yang paling berpengaruh terhadap prediksi model dengan Grad-CAM.

`find_last_conv_layer(model)`:

- mencari semua layer bertipe `nn.Conv2d`,
- mengambil layer convolution terakhir.

Grad-CAM membutuhkan layer convolution karena heatmap dibuat dari feature map spasial.

`make_gradcam(model, image_path, transform, output_path)`:

1. model masuk mode evaluasi,
2. layer convolution terakhir dicari,
3. forward hook dipasang untuk menangkap aktivasi,
4. backward hook dipasang untuk menangkap gradient,
5. gambar asli dibuka dan ditransformasi,
6. model memprediksi kelas,
7. skor kelas prediksi di-backward,
8. gradient dan aktivasi dipakai untuk membuat heatmap CAM,
9. CAM dinormalisasi,
10. CAM di-resize ke ukuran gambar asli,
11. gambar asli dan overlay heatmap ditampilkan berdampingan,
12. hasil disimpan ke `gradcam_example.png`.

Cara membaca gambar Grad-CAM:

- panel kiri: gambar MRI asli,
- panel kanan: gambar MRI dengan heatmap,
- warna panas pada heatmap menunjukkan area yang lebih memengaruhi prediksi,
- judul menunjukkan kelas yang diprediksi model.

Output run terbaru:

- Grad-CAM berhasil dibuat,
- layer yang dipakai: `stages.3.blocks.1.convs.2`.

Fungsi gambar `gradcam_example.png`:

- memberi interpretasi visual sederhana,
- membantu melihat apakah model fokus pada area yang masuk akal,
- bukan bukti medis final, tetapi alat bantu interpretasi model.

## Cell 18 - Menyimpan Konfigurasi dan Ringkasan Penelitian

Cell ini menyimpan metadata dan ringkasan eksperimen.

`experiment_config` berisi:

- lokasi project,
- lokasi dataset,
- lokasi output,
- daftar kelas,
- random seed,
- device,
- konfigurasi baseline,
- konfigurasi tuning terbaik,
- konfigurasi final,
- lokasi data eksternal,
- path laporan kualitas data,
- limitation penelitian.

Limitation yang ditulis penting:

- dataset berupa JPG 2D,
- tidak tersedia `patient_id` yang jelas,
- evaluasi dilakukan pada tingkat gambar,
- pembersihan hash hanya menghapus duplikat identik,
- pemisahan tingkat pasien belum bisa dijamin.

File yang dibuat:

- `experiment_config.json`,
- `summary.txt`.

`summary.txt` berisi:

- nama eksperimen,
- lokasi dataset,
- daftar kelas,
- informasi evaluasi tingkat gambar,
- accuracy baseline,
- accuracy final,
- macro F1 baseline,
- macro F1 final,
- lokasi folder hasil.

Fungsi cell ini: membuat dokumentasi eksperimen agar hasilnya bisa ditinjau ulang tanpa membuka seluruh notebook.

## Cell 19 - Audit Overfitting

Cell ini memeriksa indikasi overfitting berdasarkan gap macro F1 antara train dan validation.

`overfitting_summary(history_df, model_name)`:

- mencari epoch dengan validation macro F1 terbaik,
- mengambil nilai train macro F1 pada epoch terbaik,
- menghitung gap train-validation pada epoch terbaik,
- mengambil train macro F1 epoch terakhir,
- mengambil validation macro F1 epoch terakhir,
- menghitung gap train-validation pada epoch terakhir.

Hasil baseline dan final digabung menjadi DataFrame `overfitting_audit`, lalu disimpan ke:

```text
04_comparison/overfitting_audit.csv
```

Aturan kesimpulan:

- gap >= 0.10: indikasi overfitting kuat,
- gap >= 0.05: indikasi overfitting sedang,
- selain itu: tidak terlihat overfitting berat dari gap train-validation.

Output run terbaru:

- baseline selesai 15 epoch,
- model final selesai 16 epoch,
- gap terakhir baseline sekitar 0.0178,
- gap terakhir final sekitar 0.0125,
- kesimpulan: tidak terlihat overfitting berat dari gap training-validation.

Catatan cell ini juga penting:

> akurasi tinggi pada test BRISC2025 tidak membuktikan model pasti generalisasi ke rumah sakit, scanner, atau dataset lain.

Fungsi cell ini: memberi audit sederhana agar performa tinggi tidak langsung dianggap pasti generalisasi.

## Penjelasan Semua Gambar yang Dihasilkan

Berikut fungsi setiap gambar utama dari notebook.

### dataset_distribution.png

Dihasilkan oleh Cell 5. Gambar ini adalah countplot jumlah gambar per kelas untuk train dan test.

Fungsinya:

- melihat keseimbangan jumlah data antar kelas,
- melihat apakah train dan test punya distribusi yang mirip,
- membantu menilai apakah metrik accuracy bisa bias ke kelas mayoritas.

### sample_images.png

Dihasilkan oleh Cell 6. Gambar ini menampilkan contoh MRI dari setiap kelas.

Fungsinya:

- memastikan gambar terbaca,
- melihat contoh visual tiap label,
- membantu mendeteksi masalah dataset secara manual.

### loss_curve.png

Dihasilkan oleh Cell 10 setiap kali `fit_model` selesai, yaitu untuk baseline, tuning trial, dan final.

Fungsinya:

- melihat apakah train loss dan validation loss turun,
- mendeteksi tanda overfitting jika train loss turun tetapi validation loss memburuk,
- mengecek stabilitas training.

### metric_curve.png

Dihasilkan oleh Cell 10 setiap kali `fit_model` selesai.

Fungsinya:

- melihat perkembangan train accuracy,
- melihat perkembangan validation accuracy,
- melihat perkembangan validation macro F1,
- membantu memahami kapan model mencapai performa terbaik.

### confusion_matrix_test.png

Dihasilkan oleh Cell 11 saat evaluasi test.

Fungsinya:

- menunjukkan jumlah prediksi benar dan salah,
- diagonal berarti benar,
- luar diagonal berarti salah,
- membantu melihat kelas mana yang paling sering tertukar.

### correct_examples.png

Dihasilkan oleh Cell 16. Gambar ini menampilkan contoh prediksi benar model final.

Fungsinya:

- melihat contoh data yang dikenali model dengan benar,
- menampilkan label asli, label prediksi, dan confidence.

### wrong_examples.png

Dihasilkan oleh Cell 16. Gambar ini menampilkan contoh prediksi salah model final.

Fungsinya:

- menganalisis kesalahan model,
- melihat apakah kesalahan terjadi pada gambar yang ambigu,
- membantu menentukan apakah model perlu data tambahan, augmentasi, atau evaluasi lebih lanjut.

### gradcam_example.png

Dihasilkan oleh Cell 17. Gambar ini menampilkan gambar asli dan overlay Grad-CAM.

Fungsinya:

- memberi gambaran area mana yang memengaruhi keputusan model,
- membantu interpretasi model,
- memeriksa apakah model fokus pada area citra yang masuk akal.

Catatan penting: Grad-CAM adalah alat interpretasi, bukan bukti diagnosis. Heatmap menunjukkan pengaruh terhadap prediksi model, bukan lokasi tumor yang pasti secara medis.

## Hasil Historis Sebelum Pipeline Baru

Run terbaru yang terbaca berada di:

```text
E:\CNN(Tumor otak)\Code Program\hasil_edgenext\20261003_195329
```

Hasil utama run tersebut:

- accuracy baseline: 0.987,
- accuracy setelah tuning: 0.992,
- macro F1 baseline: 0.988728430559131,
- macro F1 setelah tuning: 0.9930125573532963,
- tuning meningkatkan macro F1 sekitar +0.0043,
- tidak terlihat overfitting berat berdasarkan gap train-validation,
- model tetap perlu diuji pada data eksternal karena test BRISC2025 belum menjamin generalisasi ke dataset/rumah sakit/scanner lain.

Nilai di atas adalah baseline historis. Setelah crop-pad-normalisasi dan audit eksternal baru ditambahkan, notebook harus dijalankan ulang; jangan mencampurkan angka run lama dengan hasil pipeline baru.

## Cell 20 - Audit Dataset Eksternal Mendeley dan Ukuran 224

Cell 20 memakai script:

`E:\CNN(Tumor otak)\Code Program\scripts\prepare_mendeley_external.py`

Fungsinya adalah:

1. membaca dataset Mendeley yang sudah diunduh dari `external_datasets\mendeley_82mtzd8x72_v1\raw`;
2. memetakan `Healthy` menjadi `no_tumor` dan `Pituitary Macroadenoma` menjadi `pituitary`;
3. menghitung pHash seluruh gambar eksternal dan membandingkannya dengan gambar klasifikasi BRISC;
4. menyimpan overlap pHash, overlap SHA-256, jarak pHash terdekat, status augmentasi, dan status modality;
5. membuat laporan per dataset dan per kelas;
6. menguji satu sampel internal dan satu sampel eksternal melalui transform evaluasi;
7. memastikan bentuk tensor tepat `3 x 224 x 224`.

Cell ini tidak menyatakan gambar eksternal valid sebagai T1-kontras secara otomatis. Jika arsip tidak memiliki metadata T1 dan patient/group ID, jumlah `eligible_for_strict_eval` tetap nol dan evaluasi external strict diblokir. Ini mencegah skor external dibangun dari modality yang salah atau pembagian pasien yang bocor.

### Fungsi `assert_preprocessed_size`

Fungsi ini membuka satu gambar dari sebuah DataFrame, menjalankan transform evaluasi, lalu memeriksa bahwa bentuk tensor sama dengan `(3, IMAGE_SIZE, IMAGE_SIZE)`. Karena `IMAGE_SIZE = 224`, bentuk yang diterima adalah `3 x 224 x 224`. Jika tidak sama, fungsi melempar `AssertionError` sehingga perubahan preprocessing tidak diam-diam mengubah ukuran input model.

### File hasil Cell 20

Hasil audit disimpan di folder run:

`E:\CNN(Tumor otak)\Code Program\hasil_edgenext\<RUN_ID>\05_external_mendeley_audit`

File pentingnya adalah `mendeley_manifest.csv`, `report_per_dataset_per_class.csv`, `verified_metadata_template.csv`, dan `audit_summary.json`.

