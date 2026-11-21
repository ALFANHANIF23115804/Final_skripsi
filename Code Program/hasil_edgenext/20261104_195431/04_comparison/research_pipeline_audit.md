# Audit Pipeline Penelitian EdgeNeXt

Run terbaru: `E:\CNN(Tumor otak)\Code Program\hasil_edgenext\20261104_195431`

## Status

| Pemeriksaan | Status | Detail |
|---|---|---|
| Output utama | PASS | Semua output utama tersedia |
| Tuning berjalan | PASS | 1 trial tersimpan |
| Pemilihan validation macro F1 | PASS | Model dipilih berdasarkan validation macro F1 |
| Strategi class_weight | WARN | Belum diimplementasikan sebagai eksperimen |
| Strategi focal_loss | WARN | Belum diimplementasikan sebagai eksperimen |
| Strategi oversampling | WARN | Belum diimplementasikan sebagai eksperimen |
| Variasi eksperimen image_size | WARN | Hanya satu nilai: ['224']; belum ada perbandingan faktor |
| Variasi eksperimen drop_rate | PASS | Nilai yang tersimpan: ['0.0', '0.2'] |
| Variasi eksperimen learning_rate | WARN | Hanya satu nilai: ['0.0001']; belum ada perbandingan faktor |
| Variasi eksperimen weight_decay | WARN | Hanya satu nilai: ['0.0001']; belum ada perbandingan faktor |
| Multi-seed 42/43/44 | WARN | Seed yang terlihat di source: ['42']; belum berarti sudah dijalankan |
| Seed tersimpan per konfigurasi | WARN | Belum ada random_seed di: ['E:\\CNN(Tumor otak)\\Code Program\\hasil_edgenext\\20261104_195431\\01_baseline\\config.json', 'E:\\CNN(Tumor otak)\\Code Program\\hasil_edgenext\\20261104_195431\\02_tuning\\trial_1\\config.json'] |
| Prediksi dan metrik konsisten | PASS | 1000 prediksi; accuracy=0.988, macro_f1=0.9895610677204749 |

## Kesimpulan audit

Pipeline teknis dapat membaca dataset, membuat DataLoader, membangun EdgeNeXt, melakukan training/evaluasi, dan menyimpan hasil. Namun status PASS teknis tidak berarti seluruh rancangan eksperimen lanjutan sudah selesai.

Eksperimen class weight, focal loss, oversampling, augmentasi sebagai faktor terpisah, image-size comparison, dan validasi konsistensi multi-seed harus ditambahkan atau dijalankan sebelum menyimpulkan strategi optimasi yang konsisten.
