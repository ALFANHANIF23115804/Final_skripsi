# Penjelasan Pipeline Dataset Eksternal Mendeley

File utama:

`E:\CNN(Tumor otak)\Code Program\scripts\prepare_mendeley_external.py`

Script ini adalah lapisan audit dan persiapan. Ia tidak menghapus arsip mentah dan tidak memaksa evaluasi ketika bukti T1-kontras atau patient ID tidak ada.

## Konstanta

`CLASS_NAMES` menetapkan empat label kanonik yang harus sama dengan BRISC: `glioma`, `meningioma`, `no_tumor`, dan `pituitary`.

`CLASS_ALIASES` menerjemahkan nama folder dataset Mendeley. `Healthy` menjadi `no_tumor`; `Pituitary Macroadenoma` menjadi `pituitary`.

`IMAGE_EXTENSIONS` membatasi pemindaian pada format gambar yang dapat dibaca Pillow.

`AUGMENTATION_TOKENS` berisi token seperti `aug`, `flip`, `rotate`, `noise`, dan `synthetic`. Token ini hanya memberi tanda kecurigaan berdasarkan path/nama file; ia bukan bukti sempurna bahwa gambar pernah diaugmentasi.

`MODALITY_TOKENS` mengenali petunjuk nama file seperti `t1`, `t1ce`, `contrast`, `gad`, `t2`, dan `flair`. Status `t1_candidate` bukan `t1_verified`; verifikasi tetap memerlukan metadata sumber.

## Fungsi audit file

### `file_sha256(path, chunk_size=...)`

Membaca file per potongan dan menghasilkan SHA-256. SHA berguna untuk mendeteksi salinan byte-per-byte yang identik. SHA saja tidak cukup karena resize, recompression, perubahan format, atau metadata dapat membuat isi byte berbeda walaupun visualnya sama.

### `_dct_2d(values)`

Menghitung Discrete Cosine Transform dua dimensi dengan basis kosinus NumPy. DCT mengubah citra menjadi koefisien frekuensi. Bagian frekuensi rendah menangkap struktur visual umum dan lebih tahan terhadap perubahan kecil.

### `phash_hex(path, hash_size=8, highfreq_factor=4)`

Mengubah gambar menjadi grayscale, resize ke ukuran kecil, mengambil koefisien DCT frekuensi rendah, lalu membandingkannya dengan median untuk membentuk bit hash. Hasilnya hash perceptual 63-bit yang ditulis sebagai hexadecimal. Dua gambar yang berbeda byte tetapi sangat mirip secara visual dapat memiliki jarak pHash kecil.

### `phash_distance(first, second)`

Menggunakan XOR dua hash lalu menghitung jumlah bit yang berbeda. Nilai kecil berarti kemiripan perceptual lebih tinggi. Threshold harus dicatat karena tidak ada satu threshold universal untuk semua dataset.

### `normalize_name(value)`

Menyamakan penamaan folder dengan mengubah huruf menjadi lowercase dan tanda baca menjadi underscore. Ini memungkinkan `Pituitary Macroadenoma` dipetakan secara konsisten.

### `find_class_root(root)`

Mencari folder yang anak-anaknya berisi minimal tiga nama kelas yang dikenal. Fungsi ini menangani struktur arsip bertingkat seperti `raw/.../Brain Cancer MRI Dataset/Brain Cancer MRI Dataset`.

### `modality_status(path)`

Memeriksa token nama file untuk petunjuk modality. Hasilnya:

- `t1_candidate`: ada token T1/T1CE/contrast/GAD;
- `not_t1`: ada token T2/FLAIR/DWI/ADC;
- `unknown`: tidak ada bukti pada nama file.

Fungsi ini tidak mengubah `t1_candidate` menjadi T1 terverifikasi. Itu disengaja agar model tidak diuji dengan modality yang tidak terbukti.

### `augmentation_status(path)`

Menandai path yang mengandung token augmentasi. Hasil `suspected_augmentation` harus dikeluarkan dari evaluasi atau diperiksa manual. Arsip mentah tetap dipertahankan untuk auditabilitas.

### `scan_external(root)`

Memindai folder kelas eksternal, membuka dan memverifikasi setiap gambar, menetapkan label kanonik, menghitung SHA-256 dan pHash, serta menyimpan status modality, augmentation, dan group pasien. Karena arsip Mendeley tidak menyediakan patient ID, nilai awal group adalah `unknown`.

### `scan_brisc(manifest_path)`

Membaca manifest BRISC, memilih baris task classification dan membuang mask. Pembacaan `is_mask` memakai parsing eksplisit terhadap string `true/false`, sehingga nilai CSV `False` tidak keliru dianggap benar. pHash BRISC dihitung dari file gambar yang benar-benar ada.

### `audit_phash(external, brisc, threshold=6)`

Untuk setiap gambar eksternal, mencari jarak pHash terkecil terhadap seluruh gambar BRISC. Fungsi menulis jarak terdekat, overlap SHA, overlap pHash, dan kelayakan setelah deduplikasi. Jika BRISC tidak terbaca, fungsi tidak menyatakan gambar aman; kolom audit dibuat konservatif.

## Fungsi metadata dan kelayakan

### `apply_verified_metadata(frame, metadata_path)`

Menggabungkan CSV yang diverifikasi manusia ke manifest berdasarkan `image_path`. Kolom minimalnya adalah `image_path`, `modality_status`, dan `group_id`. Baris yang tidak ada di CSV tetap `unknown`; tidak ada nilai yang dibuat-buat. `group_status` menjadi `verified_patient_id` hanya jika `group_id` benar-benar diisi.

Template yang dihasilkan script dapat diedit setelah metadata resmi diperiksa. Isi `t1_verified` hanya untuk gambar yang benar-benar terbukti T1-kontras.

### `mark_strict_eligibility(frame)`

Menetapkan `eligible_for_strict_eval=True` hanya jika semua syarat terpenuhi: tidak overlap, modality `t1_verified`, tidak terindikasi augmentasi, dan group pasien terverifikasi. Ini adalah pagar metodologis utama terhadap evaluasi yang tampak bagus tetapi bocor atau tidak sebanding.

### `make_group_kfold(frame, n_splits=5)`

Membuat fold dengan `GroupKFold`. Fungsi sengaja melempar error jika ada group `unknown` atau jumlah group unik kurang dari jumlah fold. Jadi, gambar dari pasien yang sama tidak dapat tersebar ke train dan validasi secara tidak sengaja.

## Fungsi preprocessing

### `crop_brain_pad_square(image)`

Mengubah citra ke grayscale untuk mencari area foreground, memangkas border yang dominan hitam, lalu memberi padding nol agar ukuran menjadi persegi. Tujuannya mengurangi shortcut dari rasio canvas atau border, bukan melakukan segmentasi tumor.

### `normalize_intensity(image, size=224)`

Menjalankan crop, padding, resize 224 x 224, lalu normalisasi intensitas per gambar menggunakan percentile 1 dan 99. Piksel di luar rentang dipotong ke 0 sampai 255. Normalisasi ini hanya menyamakan skala visual; ia tidak boleh dipakai untuk mengklaim peningkatan akurasi tanpa eksperimen terkontrol.

## Fungsi laporan

### `write_reports(external, output_dir)`

Menyimpan:

- `mendeley_manifest.csv`: satu baris per gambar dan seluruh status audit;
- `verified_metadata_template.csv`: template pengisian modality dan group pasien;
- `report_per_dataset_per_class.csv`: hitungan per dataset dan kelas;
- `audit_summary.json`: ringkasan jumlah, modality, dugaan augmentasi, kelayakan strict, sumber, dan status blokir.

## Fungsi `main()` dan cara menjalankan

`main()` membaca argumen command line, memindai Mendeley, menerapkan metadata verifikasi bila diberikan, membaca BRISC, menjalankan audit pHash, menghitung kelayakan strict, menulis laporan, dan mencetak ringkasan.

Dari folder `E:\CNN(Tumor otak)\Code Program`, jalankan:

```powershell
python scripts\prepare_mendeley_external.py `
  --external-root external_datasets\mendeley_82mtzd8x72_v1\raw `
  --brisc-manifest brisc2025-v6\brisc2025\manifest.csv `
  --output-dir hasil_edgenext\external_mendeley_audit
```

Setelah metadata T1 dan group pasien diverifikasi, jalankan ulang dengan tambahan:

```powershell
--verified-metadata hasil_edgenext\external_mendeley_audit\verified_metadata.csv
```

Tanpa metadata itu, hasil yang sah adalah laporan audit dan status `blocked_until_T1_and_patient_metadata_are_verified`, bukan skor external final.

## Batasan yang harus ditulis di laporan penelitian

Halaman sumber menyebut empat kelas dan menjelaskan bahwa data asli mencakup sekuens T1 dan T2. Arsip lokal yang dipasang berisi 1.137 JPG, bukan 1.200 file seperti angka ringkasan halaman. Nama file numerik tidak menyediakan modality atau patient ID. Karena itu, jumlah tersebut belum boleh dianggap sebagai jumlah T1-kontras yang eligible.

Sumber: [Mendeley Data](https://data.mendeley.com/datasets/82mtzd8x72/1).
