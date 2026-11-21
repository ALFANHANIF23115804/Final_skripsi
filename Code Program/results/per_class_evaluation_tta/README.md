# Evaluasi per kelas tumor otak

Evaluasi ini hanya melakukan inference menggunakan checkpoint yang sudah ada; tidak ada training atau perubahan arsitektur.

## Cara menjalankan

```powershell
python evaluate_per_class.py
python evaluate_per_class.py --checkpoint hasil_edgenext\20261029_182646\02_tuning\trial_1\best_model.pt
```

Output ditulis ke `E:\CNN(Tumor otak)\Code Program\results\per_class_evaluation_tta`. Checkpoint yang dievaluasi: `E:\CNN(Tumor otak)\Code Program\hasil_edgenext\20261104_195431\01_baseline\best_model.pt`.

## Dataset, kelas, dan preprocessing

- Dataset: `E:\CNN(Tumor otak)\Code Program\brisc2025-v6\brisc2025\classification_task`
- Split yang dievaluasi: folder `test` resmi, 1000 gambar (glioma: 254, meningioma: 306, no_tumor: 140, pituitary: 300)
- Kelas dari checkpoint/dataset: glioma, meningioma, no_tumor, pituitary
- Preprocessing inference sama dengan notebook: crop foreground, pad menjadi persegi, resize 224, grayscale percentile 1–99, convert RGB, dan ImageNet normalization.
- Model: EdgeNeXt dari `timm`; bobot dimuat dari `model_state_dict`, tanpa pretrained download ulang.

## Metrik

Per kelas dihitung precision, recall/sensitivity, specificity one-vs-rest dari TP/TN/FP/FN, F1-score, support, ROC-AUC OVR, dan PR-AUC. Ringkasan juga mencakup macro/weighted average, balanced accuracy, dan multiclass Matthews Correlation Coefficient.

## Hasil setiap kelas

| class | precision | recall_sensitivity | specificity | f1_score | support | roc_auc_ovr | pr_auc_ovr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glioma | 0.9843 | 0.9882 | 0.9946 | 0.9862 | 254 | 0.9989 | 0.9978 |
| meningioma | 0.9868 | 0.9804 | 0.9942 | 0.9836 | 306 | 0.9983 | 0.9934 |
| no_tumor | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 140 | 1.0000 | 1.0000 |
| pituitary | 0.9900 | 0.9933 | 0.9957 | 0.9917 | 300 | 0.9994 | 0.9990 |

## Ringkasan keseluruhan

- Accuracy: 0.9890
- Macro F1-score: 0.9904
- Balanced accuracy: 0.9905
- MCC: 0.9850
- ROC-AUC OVR macro: 0.9992
- PR-AUC OVR macro: 0.9976
- Weighted F1-score: 0.9890

## Kelas paling sulit

Berdasarkan gabungan recall, F1, dan specificity terendah, kelas yang paling perlu diperhatikan adalah **meningioma**. Recall terendah: **meningioma** (0.9804); F1 terendah: **meningioma** (0.9836).

Pasangan tertukar terbanyak adalah **meningioma → glioma** sebanyak **4** gambar. Rincian ada di `confusion_pairs.csv` dan folder `misclassified_by_pair/`.

## Analisis kesalahan dan Grad-CAM

`misclassified_examples.png` menampilkan contoh salah; `low_confidence_examples.png` menampilkan 20 confidence terendah; `misclassified_by_pair/` mengelompokkan kesalahan berdasarkan pasangan kelas; dan `gradcam/` berisi contoh benar, salah, serta confidence rendah jika layer Conv2d model mendukung Grad-CAM.

## Integritas dan keterbatasan

Audit SHA-256 train–test dicatat di `data_leakage_audit.csv`; overlap exact pada folder mentah: 7, overlap pada training efektif setelah deduplikasi test-hash: 0, baris train dikeluarkan karena hash test: 9, duplikat internal train tambahan: 36, sehingga referensi input training efektif menjadi 4955 baris. Patient/group columns yang terdeteksi: `not_available`. Dataset ini tidak menyediakan identitas pasien yang jelas pada manifest, sehingga pemisahan berbasis pasien tidak dapat dibuktikan. Audit image-level tidak menjamin generalisasi ke pasien, scanner, atau rumah sakit baru.

## Rekomendasi penelitian selanjutnya

Gunakan `experiment_comparison.csv` dan jalankan satu perubahan per eksperimen: class weight, focal loss, augmentation, oversampling minoritas, CLAHE/contrast, kemudian ukuran citra. Pilih model berdasarkan validation macro F1, lalu laporkan test macro F1, balanced accuracy, recall tiap kelas, dan interval ketidakpastian bila memungkinkan. Tambahkan patient ID dan lakukan patient-level split pada rilis data berikutnya.
