# Analisis Kesalahan Pasangan Kelas EdgeNeXt

Run: `E:\CNN(Tumor otak)\Code Program\hasil_edgenext\20261104_195431`

## Ringkasan hasil nyata

- Baseline accuracy: `0.988`
- Final accuracy: `0.988`
- Baseline macro F1: `0.9893908010308982`
- Final macro F1: `0.9895610677204749`
- Total kesalahan final: `12` dari `1000` gambar

## Pasangan kelas yang tertukar

| Kelas sebenarnya | Prediksi | Jumlah | Confidence rata-rata |
|---|---|---:|---:|
| glioma | meningioma | 3 | 0.7692 |
| meningioma | glioma | 3 | 0.8458 |
| meningioma | pituitary | 3 | 0.7005 |
| pituitary | meningioma | 2 | 0.7286 |
| pituitary | glioma | 1 | 0.6001 |

## Interpretasi

- Pasangan dengan jumlah error terbesar menjadi prioritas analisis visual dan eksperimen optimasi.
- Confidence tinggi pada prediksi salah menunjukkan kesalahan yang lebih sistematis; confidence rendah menunjukkan contoh ambigu atau sulit.
- Distribusi orientasi hanya bersifat indikatif karena orientasi diperoleh dari token nama file.
- Grad-CAM dapat digunakan untuk melihat area yang memengaruhi model, tetapi bukan bukti segmentasi tumor.

## Status konsistensi optimasi

Hasil ini belum membuktikan konsistensi lintas random seed. Konsistensi harus diuji dengan menjalankan konfigurasi terbaik pada beberapa seed dan melaporkan rata-rata serta standar deviasi macro F1, recall kelas sulit, dan jumlah error pasangan.

## Artefak

- `misclassified_detailed.csv`: seluruh prediksi salah dengan confidence dan orientasi.
- `error_pairs_detailed.csv`: ringkasan error per pasangan kelas.
- `error_pairs_by_plane.csv`: error per pasangan dan orientasi.
- `error_confidence_bands.csv`: error berdasarkan rentang confidence.
